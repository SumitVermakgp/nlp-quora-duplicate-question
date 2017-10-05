import tensorflow as tf
import os
import random
import time
import numpy as np
import prefetch_queue
import csv
import json
import itertools

from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS
DATASET_SIZE = 404292

tf.app.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,"""Whether to log device placement.""")
tf.app.flags.DEFINE_string('logdir', None, """Where to save logs, and whether to""")
tf.app.flags.DEFINE_string("results_dir", None, "Where to store results")

tf.app.flags.DEFINE_boolean('debug', False, "debug")
tf.app.flags.DEFINE_boolean('restore', False, "Skip training and restore last saved model")
tf.app.flags.DEFINE_boolean('model_selection', False, "Start model selection")

tf.app.flags.DEFINE_string('dictionary', 'glove.840B.300d', "dictionary to use")
tf.app.flags.DEFINE_boolean('simple_network', True, """Whether to use a simple encoder network""")
tf.app.flags.DEFINE_boolean('reweight_classes', True, "Whether to reweight class to target test distribution")

tf.app.flags.DEFINE_integer('earlystop_checks', 10, "Number of checks (50it) to wait before expecting an average performance improvement")
tf.app.flags.DEFINE_float('lr', 0.001, """Learning rate""")
tf.app.flags.DEFINE_float('adameps', 1e-8, "Epsilon of Adam optimizer")
tf.app.flags.DEFINE_integer('trainbatch', 1024, "Training batch size")
tf.app.flags.DEFINE_float('dropout_rate', 0.0, "Probability of a neuron being dropped out")

tf.app.flags.DEFINE_integer('question_embedder_size', 250, "RNN size")

tf.app.flags.DEFINE_integer('se_readout_size', 125, 'Read out size')

tf.app.flags.DEFINE_integer('nr_max_iteration', 5, '')
tf.app.flags.DEFINE_integer('nr_attention_hidden_size', 100, '')
tf.app.flags.DEFINE_integer('nr_reasoner_state_size', 300, '')

class DuplicateQuestionDetector:
    class NetworkOption:
        def __init__(self):
            self.word_embedding_size = None
            self.question_embedder_size = 100

    class SimpleNetworkOption(NetworkOption):
        def __init__(self):
            DuplicateQuestionDetector.NetworkOption.__init__(self)
            # all the option are in FLAGS

    class ReasonerOption(NetworkOption):
        def __init__(self):
            DuplicateQuestionDetector.NetworkOption.__init__(self)

            self.attention_hidden_size = 50

            self.reasoner_state_size = 100
            self.max_reasoner_iteration = 5
            self.reasoner_hidden_size = 110

    class TrainOption:
        def __init__(self):
            self.selection_batch_size = 1000

            self.data_dir = None
            self.dictionary_path = None
            self.log_device_placement = True
            
    def __init__(self, option=NetworkOption()):
        self.opt = option

    @staticmethod
    def _get_input_from_ids_t(input_files, dict_t, has_label=True, max_epoch=None):
        filename_queue = tf.train.string_input_producer(input_files, num_epochs=max_epoch)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        context_features = {
            'id': tf.FixedLenFeature([], tf.int64),
            'length1': tf.FixedLenFeature([], tf.int64),
            'length2': tf.FixedLenFeature([], tf.int64),
            'extra_feature': tf.FixedLenFeature([1], tf.float32),
        }

        if has_label:
            context_features['is_duplicate'] = tf.FixedLenFeature([], tf.int64)

        input_tensors = tf.parse_single_sequence_example(
            serialized_example,
            context_features=context_features,
            sequence_features={
                "question1": tf.FixedLenSequenceFeature([1], tf.int64),
                "question2": tf.FixedLenSequenceFeature([1], tf.int64)
            })

        input_tensors[1]["question1"] = tf.gather(dict_t, tf.squeeze(input_tensors[1]['question1'], axis=1))
        input_tensors[1]["question2"] = tf.gather(dict_t, tf.squeeze(input_tensors[1]['question2'], axis=1))

        return input_tensors

    @staticmethod
    def _get_batch_t(input_tensors, batch_size):

        tensors = [
            input_tensors[1]["question1"],
            input_tensors[1]["question2"],
            tf.cast(input_tensors[0]["length1"], tf.int32),
            tf.cast(input_tensors[0]["length2"], tf.int32)
            ]

        if 'is_duplicate' in input_tensors[0]:
            tensors.append(input_tensors[0]["is_duplicate"])

        tensors.append(input_tensors[0]['id'])
        tensors.append(input_tensors[0]['extra_feature'])

        # TODO shuffle ?
        return tf.train.batch(tensors,
            batch_size, dynamic_pad=True, num_threads=8, capacity=batch_size*10, allow_smaller_final_batch=True)

    def _question_embedder(self, question_t, length_t, rstate_t, reuse=False, is_training=False):
        """
        bidirectional embedding with lstm, then attention mixing of embeddings.
        :param question_t: [ batch_size, max_length, word_embedding_size ]
        :param length_t: [ batch_size ]
        :param rstate_t: [ batch_size, reasoner_state_size ]
        :param reuse:
        :return:
        """

        with tf.variable_scope("QuestionEmbedder", reuse=reuse) as vs:
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.opt.question_embedder_size)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.opt.question_embedder_size)

            if is_training:
                p = 1.0 - FLAGS.dropout_rate
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, p, p, p)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, p, p, p)

            (outs_fw, outs_bw), \
            (states_fw, states_bw) = \
                tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, question_t, length_t,
                                                           dtype= tf.float32,
                                                           time_major= False, swap_memory=True, parallel_iterations=64)   # TODO optimize (swap mem, parallel its)

            outs = tf.concat((outs_fw, outs_bw), 2, name='outs') # assuming [ batch_size, max_length, out_size]
            states = tf.concat((states_fw, states_bw), 2, name='states')

            # TODO check tf.contrib.rnn.AttentionCellWrapper()
            with tf.variable_scope("Attention"):

                with tf.variable_scope("Data"):
                    rstates = tf.tile(tf.expand_dims(rstate_t, axis=1), [1, tf.shape(outs)[1], 1])  # tile along question lenght dimension
                    alldata = tf.concat([outs, rstates], axis=2)
                    all_batch = tf.reshape(alldata, [-1, self.opt.question_embedder_size * 2 + self.opt.reasoner_state_size])

                with tf.variable_scope("Net"):
                    att_h = tf.layers.dense(all_batch, self.opt.attention_hidden_size)
                    att_o = tf.layers.dense(att_h, 1, tf.sigmoid)
                att = tf.reshape(att_o, [tf.shape(question_t)[0], -1, 1])

                att_masked = tf.multiply(att, tf.expand_dims(tf.sequence_mask(length_t, dtype=tf.float32), -1))

                with tf.variable_scope("AttentionCoef"):
                    summed = tf.reduce_sum(att_masked, axis=1, keep_dims=True)
                    tiled = tf.tile(summed, [1, tf.shape(att_masked)[1],1])
                    scaling = att_masked / (tiled + 1e-14)
                delta_q = tf.reduce_sum(tf.multiply(outs, scaling), axis=1)

                return delta_q, scaling

    def _reasoner(self, q1d_t, q2d_t, rstate_t, extra_features_t, reuse=False, is_training=False):
        with tf.variable_scope("Reasoner", reuse=reuse):
            x = tf.concat([q1d_t, q2d_t, extra_features_t], axis=1)
            cell = tf.nn.rnn_cell.LSTMCell(self.opt.reasoner_state_size/2, reuse=reuse, state_is_tuple=False)

            if is_training:
                p = 1.0 - FLAGS.dropout_rate
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,p,p,p)

            out, state = cell(x, rstate_t)
            return (out, state)

    def _answerer(self, rstate_t, reuse=False):
        with tf.variable_scope('Answerer', reuse=reuse):
            a = tf.layers.dense(rstate_t, 2, activation=tf.sigmoid)
            out, is_ready = a[:,0], tf.greater_equal(a[:,1], 0.5)
            return out, is_ready

    def network_simple(self, q1_t, q2_t, l1_t, l2_t, extra_features_t, reuse=False, is_training=True, sample_attention=False):
        """
        :param q1_t: [ batch_size, padded_length, word_embedding_size ]
        :param q2_t: [ batch_size, padded_length, word_embedding_size ]
        :param l1_t: [ batch_size ]
        :param l2_t: [ batch_size ]
        :param extra_features_t: [ batch_size ]
        :return:
        """

        with tf.variable_scope("Network", reuse=reuse):

            with tf.variable_scope("QuestionEmbedder", reuse=reuse) as vs:
                cell = tf.nn.rnn_cell.LSTMCell(self.opt.question_embedder_size)

                if is_training:
                    p = 1.0 - FLAGS.dropout_rate
                    cell = tf.contrib.rnn.DropoutWrapper(cell, p, p, p)

                outs1, states1 = tf.nn.dynamic_rnn(cell, q1_t, l1_t, dtype=tf.float32, time_major=False,
                                                   swap_memory=True,
                                                   parallel_iterations=128)
                vs.reuse_variables()

                outs2, states2 = tf.nn.dynamic_rnn(cell, q2_t, l2_t, dtype=tf.float32, time_major=False,
                                                   swap_memory=True,
                                                   parallel_iterations=128)

                batch_n = tf.range(start=0, limit=tf.shape(q1_t)[0])
                outs2 = tf.gather_nd(outs2, tf.stack([batch_n, l2_t-1], axis=1))
                outs1 = tf.gather_nd(outs1, tf.stack([batch_n, l1_t-1], axis=1))

            with tf.variable_scope("Reasoner", reuse=reuse):
                inp = tf.concat([outs1, outs2, extra_features_t], 1)
                inp = tf.layers.batch_normalization(inp, trainable=False)
                inp = tf.layers.dropout(inp, FLAGS.dropout_rate, training=is_training)

                hid = tf.layers.dense(inp, FLAGS.se_readout_size, activation=tf.nn.tanh)
                hid = tf.layers.batch_normalization(hid, trainable=False)
                hid = tf.layers.dropout(hid, FLAGS.dropout_rate, training=is_training)

                out = tf.layers.dense(hid, 1, activation=tf.sigmoid)

            if sample_attention:    # Just to uniform the signature
                return out, None, None
            else:
                return out

    def network(self, q1_t, q2_t, l1_t, l2_t, extra_features, reuse=False, is_training=True, sample_attention = False):
        """
        :param reuse:
        :param q1_t: [ batch_size, padded_length, word_embedding_size ]
        :param q2_t: [ batch_size, padded_length, word_embedding_size ]
        :return:
        """

        with tf.variable_scope("batch_size"):
            batch_size_t = tf.shape(q1_t)[0]

        with tf.variable_scope("Network", reuse=reuse):
            rstate0 = tf.zeros([batch_size_t, self.opt.reasoner_state_size], name="reasoner_state_0")

            self.r=reuse

            def cond(count, is_ready, rstate, outs, stats, att):
                return tf.reduce_any(tf.logical_and(
                    tf.less(count, self.opt.max_reasoner_iteration),
                    tf.logical_not(is_ready)))

            def body(count, is_ready, rstate, outs, stats, att):
                count += 1
                q1d, att1 = self._question_embedder(q1_t, l1_t, rstate, reuse=self.r, is_training=is_training)
                q2d, att2 = self._question_embedder(q2_t, l2_t, rstate, reuse=True, is_training=is_training)
                rout, rstate = self._reasoner(q1d, q2d, rstate, extra_features, reuse=self.r, is_training=is_training)

                new_outs, new_is_ready = self._answerer(rout, self.r)

                stats += tf.cast(tf.logical_not(is_ready), tf.float32)   # number of iterations

                # propagate outs until first ready
                outs = tf.multiply(tf.cast(tf.logical_not(is_ready), tf.float32), new_outs) \
                       + tf.multiply(tf.cast(is_ready, tf.float32), outs)
                is_ready = tf.logical_or(is_ready, new_is_ready)

                if sample_attention:
                    new_col = tf.concat([tf.squeeze(att1, axis=2), tf.squeeze(att2, axis=2)], axis=1)
                    prev = att[:, :count-1, :]
                    next = att[:, count:, :]
                    att = tf.concat([prev, tf.expand_dims(new_col, axis=1), next], axis=1)
                    att.set_shape([None, self.opt.max_reasoner_iteration, None])

                self.r = True
                return count, is_ready, rstate, outs, stats, att

            (count, is_ready, rstate, outs, stats) = \
                0, \
                tf.cast(tf.zeros([batch_size_t]), tf.bool), \
                rstate0, \
                tf.zeros([batch_size_t]), \
                tf.zeros([batch_size_t])

            att = tf.zeros([batch_size_t, self.opt.max_reasoner_iteration, tf.shape(q1_t)[1] + tf.shape(q2_t)[1]]) if sample_attention else 0

            (count, is_ready, rstate, outs, stats, att) = tf.while_loop(cond, body, (count, is_ready, rstate, outs, stats, att), parallel_iterations=1, swap_memory=True)


        # just one
        if not reuse:
            tf.summary.histogram("reasoner_iteration", stats)

        if sample_attention:
            return outs, att, stats
        else:
            return outs

    @staticmethod
    def _loss(y_pred, y_trg):

        # blind test labels distribution is different from labelled data
        # see https://www.kaggle.com/c/quora-question-pairs/discussion/31179#173868
        # reweight loss function to massimize test performance

        w = (1.309028344, 0.472001959) if FLAGS.reweight_classes else (1.0, 1.0)

        with tf.variable_scope("Loss"):

            if y_trg.dtype != tf.float32:
                y_trg = tf.cast(y_trg, tf.float32)

            eps = 1e-7    # with smaller eps: 1.0 - eps = 1.0 once converted to a float32 tensor
            y_pred, y_trg = tf.clip_by_value(tf.squeeze(y_pred), eps, 1.0 - eps), tf.clip_by_value(tf.squeeze(y_trg), eps, 1.0-eps)

            return - tf.reduce_mean(tf.multiply(y_trg, tf.log(y_pred))*w[1] + w[0] * tf.multiply((1 - y_trg), tf.log(1 - y_pred)))

    def _average_gradients(self, tower_grads):
        """
        https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101

        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # # now using tf.parallel_stack
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)

            # grads = zip(*grad_and_vars)[0]
            # grad = tf.parallel_stack(grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def eval_test(self, data_dir, dictionary_path):
        network = self.network_simple if FLAGS.simple_network else self.network
        eval_batch_size = 8192 if FLAGS.simple_network else 4096

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            print "Loading dictionary..."
            dictionary = np.load(dictionary_path)
            dict_ph = tf.placeholder(tf.float32, shape=np.shape(dictionary), name="dictionary")
            dict_t = tf.Variable(dict_ph, trainable=False)
            print "Loaded dictionary"

            with tf.variable_scope("TS_DATA"):
                inp_ts = DuplicateQuestionDetector._get_input_from_ids_t([os.path.join(data_dir, "test_files/test_0.tfr")], dict_t, has_label=False, max_epoch=1)
                ts_data = DuplicateQuestionDetector._get_batch_t(inp_ts, eval_batch_size)
                ts_data_q = prefetch_queue.prefetch_queue(ts_data, dynamic_pad=True, capacity=32, num_threads=8)

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device('/gpu:0'):
                    with tf.name_scope('%s_%d' % ("Tower", 0)) as scope:
                        (q1, q2, l1, l2, ids, extra) = ts_data_q.dequeue(name='ts_batch')
                        is_duplicate = network(q1, q2, l1, l2, extra, reuse=False, is_training=False)

            saver = tf.train.Saver()

            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True
                    # ,log_device_placement=True
            )) as sess:

                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()],
                         feed_dict={dict_ph: dictionary})

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                saver.restore(sess, os.path.join(FLAGS.results_dir, "model.ckpt"))

                with open(os.path.join(FLAGS.results_dir, "test_outs.csv"), 'wb') as fout:
                    writer = csv.DictWriter(fout, ['test_id', 'is_duplicate'])
                    writer.writeheader()

                    try:
                        for c in itertools.count():
                            pred, test_id = sess.run([tf.squeeze(is_duplicate), ids])
                            if c % 20 == 0:
                                print "Evaluation %.0f%%" % (100 * eval_batch_size * c / float(2345796))

                            writer.writerows([{'test_id': i, 'is_duplicate': o} for (o,i) in zip(pred, test_id)])
                    except Exception, e:
                        # TODO what the right exception ?
                        print "Except!"
                        coord.request_stop(e)

                    # empty embeddings, just random
                    with open(os.path.join(data_dir, "test_oovstat.json")) as finfo:
                        info = json.load(finfo)

                        for i in info['empty_ids']:
                            writer.writerow({'test_id': i, 'is_duplicate': random.uniform(0.0, 1.0)})

                coord.request_stop()
                coord.join(threads)

    def train_gpus(self, opt=TrainOption()):
        files = [os.path.join(opt.data_dir, "train_files", x) for x in os.listdir(os.path.join(opt.data_dir, "train_files")) if x[0] != '.']
        random.shuffle(files)
        tr_files = files[:-1]
        sl_files = files[-1:]

        network = self.network_simple if FLAGS.simple_network else self.network

        if FLAGS.results_dir and  os.path.exists(os.path.join(FLAGS.results_dir, "info.json")):
            with open(os.path.join(FLAGS.results_dir, "info.json")) as f:
                results_info = json.load(f)
        else:
            results_info = {
                "selection_loss": 1.0
            }

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            print "Loading dictionary..."
            dictionary = np.load(opt.dictionary_path)
            dict_ph = tf.placeholder(tf.float32, shape=np.shape(dictionary), name="dictionary")
            dict_t = tf.Variable(dict_ph, trainable=False)

            with tf.variable_scope("TR_DATA"):
                inp_tr = DuplicateQuestionDetector._get_input_from_ids_t(tr_files, dict_t)
                tr_data = DuplicateQuestionDetector._get_batch_t(inp_tr, FLAGS.trainbatch) # (q1_tr, q2_tr, l1_tr, l2_tr, y_tr_trg)
                tr_data_q = prefetch_queue.prefetch_queue(tr_data, dynamic_pad=True, num_threads=4)

            with tf.variable_scope("SL_DATA"):
                inp_sl = DuplicateQuestionDetector._get_input_from_ids_t(sl_files, dict_t)
                sl_data = DuplicateQuestionDetector._get_batch_t(inp_sl, opt.selection_batch_size)
                sl_data_q = prefetch_queue.prefetch_queue(sl_data, dynamic_pad=True, num_threads=4)

            optimizer = tf.train.AdamOptimizer(FLAGS.lr, epsilon=FLAGS.adameps)

            tower_grads = []
            summaries=[]
            with tf.variable_scope(tf.get_variable_scope()):
                reuse=False
                for i in xrange(FLAGS.num_gpus):        # use CUDA_VISIBLE_DEVICES=2,3 to mask only to
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ("Tower", i)) as scope:
                            (q1_tr, q2_tr, l1_tr, l2_tr, y_tr_trg, _, extra_tr) = tr_data_q.dequeue(name='tr_batch')
                            (q1_sl, q2_sl, l1_sl, l2_sl, y_sl_trg, id_sl, extra_sl) = sl_data_q.dequeue(name='sl_batch')

                            y_tr_out = network(q1_tr, q2_tr, l1_tr, l2_tr, extra_tr, reuse, is_training=True)
                            loss_tr = DuplicateQuestionDetector._loss(y_tr_out, y_tr_trg)

                            tower_grads.append(optimizer.compute_gradients(loss_tr))

                            y_sl_out, sl_att, sl_it = network(q1_sl, q2_sl, l1_sl, l2_sl, extra_sl, reuse=True, is_training=False, sample_attention=True)
                            loss_sl = DuplicateQuestionDetector._loss(y_sl_out, y_sl_trg)

                            # For next GPUs
                            reuse = True

                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            grads = self._average_gradients(tower_grads) if len(tower_grads) > 1 else tower_grads[0]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
                train_op = apply_gradient_op

            # Add histograms for gradients.
            added_vars = set()
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            # Add histograms for trainable variables.
                for var in tf.trainable_variables():
                    if var.op.name not in added_vars:   # avoid multiple histograms
                        summaries.append(tf.summary.histogram(var.op.name, var))
                        added_vars.add(var.op.name)

            # Loss (last tower)
            summaries.append(tf.summary.scalar("TR_LOSS", loss_tr))
            summaries.append(tf.summary.scalar("SL_LOSS", loss_sl))

            sl_confusion = tf.confusion_matrix(y_sl_trg, tf.round(y_sl_out))

            # TODO remove just to debug performance
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()

            summary_op = tf.summary.merge_all()     # tf.summary.merge(summaries)
            saver = tf.train.Saver()

            with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True
                    # ,log_device_placement=True
            )) as sess:

                if FLAGS.debug:
                    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

                if FLAGS.logdir:
                    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()], feed_dict={dict_ph: dictionary})

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # early stop
                sl_loss_values = [1000 for _ in range(FLAGS.earlystop_checks * 2)]
                best_ls = 1000.0

                # speed
                t0 = time.time()
                avg_spd = 0
                global DATASET_SIZE

                for step in xrange(int(0)):
                    
                    sess.run([train_op])
                    if step % 50 == 0 and step > 0:
                        spd = float(FLAGS.num_gpus * FLAGS.trainbatch * 50) / (time.time() - t0)
                        n = int(step / 50) - 1
                        avg_spd = ((avg_spd * n) + spd) / (n+1)
                        epochs = step * FLAGS.num_gpus * FLAGS.trainbatch / float(DATASET_SIZE)
                        lt, ls = sess.run([loss_tr, loss_sl])
                        print 'step %d\t %.2f\t%.2f\t[%.0f (%.0f) sent/sec ] %.2fe' % (step, lt, ls, spd, avg_spd, epochs),
                        t0 = time.time()
                        sl_loss_values += [ls]
                        sl_loss_values = sl_loss_values[1:]

                        if FLAGS.results_dir and ls < results_info['selection_loss'] and ls < best_ls:
                            saver.save(sess, os.path.join(FLAGS.results_dir, "model.ckpt"))

                        best_ls = min([best_ls, ls])

                        # Earlystop
                        mean1 = np.mean(sl_loss_values[:FLAGS.earlystop_checks])
                        mean2 = np.mean(sl_loss_values[FLAGS.earlystop_checks:])
                        print " improvement %.2f -> %.2f" % (mean1, mean2)
                        if round(1000*mean1) <= round(1000*mean2) :
                            print "STOP: no improvement.\t best selection %.2f" % best_ls
                            break
                        if np.isnan(mean1) or np.isnan(mean2):
                            print "STOP: NaN\t  best selection %.2f" % best_ls
                            break

                    if FLAGS.logdir and step % 50 == 0 and step > 0:
                        s, conf = sess.run([summary_op, sl_confusion])     # options=run_options, run_metadata=run_metadata)
                        summary_writer.add_summary(s, global_step=step)
                        # summary_writer.add_run_metadata(run_metadata, "step%d" % step)
                        print conf,
                        print "ACC. %.1f%%" % (100 * (conf[0][0]+conf[1][1]) / float(np.sum(conf)))

                if best_ls < results_info['selection_loss'] and FLAGS.results_dir:
                    print "New best model stored"

                    info = {
                        "selection_loss": float(best_ls),
                        "FLAGS": FLAGS.__dict__
                    }

                    if not FLAGS.simple_network:
                        saver.restore(sess, os.path.join(FLAGS.results_dir, "model.ckpt"))
                        a,b,c,d,e,f,g = sess.run([y_sl_out, sl_att, y_sl_trg, id_sl, sl_it, l1_sl, l2_sl])
                        info['attention_sample'] = {
                            'pred': a.tolist(),
                            'att': b.tolist(),
                            'trg': c.tolist(),
                            'id':d.tolist(),
                            'iters': e.tolist(),
                            'l1': f.tolist(),
                            'l2': g.tolist()
                        }
                    with open(os.path.join(FLAGS.results_dir, "info.json"), 'wb') as finfo:
                        json.dump(info, finfo)

                coord.request_stop()
                coord.join(threads)

                return best_ls


def main(argv=None):

    if 'CUDA_VISIBLE_DEVICES' in os.environ and len(os.environ['CUDA_VISIBLE_DEVICES']) > 0:
        FLAGS.num_gpus = len(json.loads("["+os.environ['CUDA_VISIBLE_DEVICES']+"]"))

    DQD = DuplicateQuestionDetector
    t = DQD.TrainOption()

    if FLAGS.simple_network:
        o = DQD.SimpleNetworkOption()
    else:
        o = DQD.ReasonerOption()
        o.max_reasoner_iteration = FLAGS.nr_max_iteration
        o.attention_hidden_size = FLAGS.nr_attention_hidden_size
        o.reasoner_state_size = FLAGS.nr_reasoner_state_size

    o.word_embedding_size = 300
    o.question_embedder_size = FLAGS.question_embedder_size

    t.data_dir = "data/ids/"+FLAGS.dictionary
    t.dictionary_path = "data/"+FLAGS.dictionary+"/"+FLAGS.dictionary+".npy"
    t.selection_batch_size = 2048

    dqd = DQD(o)

    if FLAGS.restore:
        dqd.eval_test(t.data_dir, t.dictionary_path)
    else:
        dqd.train_gpus(t)


if __name__ == "__main__":
    tf.app.run()
