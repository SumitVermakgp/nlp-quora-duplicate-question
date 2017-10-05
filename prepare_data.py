import csv
import json
import re
import itertools as it
import numpy as np
import tensorflow as tf
import random
import os

ALLOWED_CHARACTERS = r"a-z0-9\^\!\'\+\-\="


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dictionary_name', "glove.840B.300d", """word vectors""")
tf.app.flags.DEFINE_boolean('train', True, 'whether to process also train set')
tf.app.flags.DEFINE_boolean('test', False, 'whether to process also test set')


def prepare_tokens(text):
    global ALLOWED_CHARACTERS

    text = text.lower()

    # some cleaning ( started from https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text)
    text = re.sub(r"pok\xc3\xa9mon", "pokemon", text)
    text = re.sub(r"[^" + ALLOWED_CHARACTERS + "]", " ", text)    # TODO consider non ASCII letters ?
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm ", "i am ", text)
    text = re.sub(r"i m ", "i am ", text)
    text = re.sub(r" hav ", " have ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"0s", "0", text)
    text = re.sub(r"0rs ", "0 rupees ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" u s a ", " america ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" u k ", " uk ", text)
    text = re.sub(r" e - mail ", " email ", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r" upvote", " up vote", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"quorans", "quora users", text)

    return text.split()


def text_to_ids_tfr(source_filepath, words_dict, target_filepath, oov_stat_path=None, symmetric=False, split_in_n_files=1, shuffle=True):

    d = "/".join(target_filepath.split('/')[:-1])
    if not os.path.exists(d):
        os.makedirs(d)

    oov_stat = {
        'frequencies': {},  # word -> freq
        'perSentence': [],  # (n oov, sent. lenght)
        'shared': [],  # ([shared oov], n oov 1, noov2, lenght 1, lenght 2)
        'empty_ids': []
    }

    with open(source_filepath) as fin:
        records = []
        count, empty, failed = 0, 0, 0

        for row in csv.DictReader(fin):
            try:
                q1 = prepare_tokens(row["question1"])
                q2 = prepare_tokens(row["question2"])

                ids1 = [words_dict[w] for w in q1 if words_dict.has_key(w)]
                ids2 = [words_dict[w] for w in q2 if words_dict.has_key(w)]

                oov1 = [w for w in q1 if not words_dict.has_key(w)]
                oov2 = [w for w in q2 if not words_dict.has_key(w)]

                for w in it.chain(oov1, oov2):
                    oov_stat['frequencies'][w] = oov_stat['frequencies'].get(w, 0) + 1

                if len(oov1):
                    oov_stat['perSentence'].append((len(oov1), len(row['question1'])))

                if len(oov2):
                    oov_stat['perSentence'].append((len(oov2), len(row['question2'])))

                oov_jaccard = 1.0
                if len(oov1) or len(oov2):
                    oov_jaccard = len(set(oov1).intersection(set(oov2))) / float(len(set(oov1).union(set(oov2))))
                    oov_stat['shared'].append(oov_jaccard)

                if len(ids1) > 0 and len(ids2) > 0:

                    q1 = [tf.train.Feature(int64_list=tf.train.Int64List(value=[i])) for i in ids1]
                    q2 = [tf.train.Feature(int64_list=tf.train.Int64List(value=[i])) for i in ids2]

                    feature = {
                        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row["id"] if row.has_key("id") else int(row["test_id"]))])),
                        'length1': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(ids1)])),
                        'length2': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(ids2)])),
                        'extra_feature': tf.train.Feature(float_list=tf.train.FloatList(value=[oov_jaccard]))
                    }

                    if row.has_key('is_duplicate'):
                        feature['is_duplicate']= tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[int(row["is_duplicate"])]))

                    record1 = tf.train.SequenceExample(
                        context=tf.train.Features(feature=feature),
                        feature_lists=tf.train.FeatureLists(
                            feature_list={
                                'question1': tf.train.FeatureList(feature=q1),
                                'question2': tf.train.FeatureList(feature=q2)
                            }
                        )
                    ).SerializeToString()

                    if symmetric:
                        t = feature['length2']
                        feature['length2'] = feature['length1']
                        feature['length1'] = t
                        record2 = tf.train.SequenceExample(
                            context=tf.train.Features(feature=feature),
                            feature_lists=tf.train.FeatureLists(
                                feature_list={
                                    'question1': tf.train.FeatureList(feature=q2),
                                    'question2': tf.train.FeatureList(feature=q1)
                                }
                            )
                        ).SerializeToString()

                        records.append((record1, record2))
                    else:
                        records.append(record1)

                    count += 1

                    if count % 1000 == 0:
                        print count
                else:
                    oov_stat['empty_ids'].append(int(row["id"] if row.has_key("id") else int(row["test_id"])))
                    empty += 1
                    print "empty"

            except:
                failed += 1
                print "failed"

        if shuffle:
            random.shuffle(records)

        chunk_size = len(records) / split_in_n_files
        chunks = [records[i*chunk_size:(i+1)*chunk_size] for i in range(split_in_n_files)]
        chunks[-1] = records[(split_in_n_files-1) * chunk_size:]

        for n in range(split_in_n_files):
            writer = tf.python_io.TFRecordWriter(target_filepath + "_%d.tfr" % n)

            # prevent same pairs to be both in train and selection (swapped)
            if symmetric:
                ab, ba = zip(*(chunks[n]))
                c = [_ for _ in ab + ba]
                random.shuffle(c)
            else:
                c = chunks[n]
            for r in c:
                writer.write(r)

        print "Failed: %d\tEmpty: %d\tDone: %d" % (failed, empty, count)

        if oov_stat_path:
            with open(oov_stat_path, 'wb') as fout:
                json.dump(oov_stat, fout)


def store_embeddings_binary(source, target, strip_not_matching=None):

    with open(source, 'rb') as f:
        csv.field_size_limit = 2 ** 20
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)

        if strip_not_matching:
            d = [(row[0], map(float, row[1:])) for row in reader if re.match(strip_not_matching, row[0]) and len(row[0]) > 0]
        else:
            d = [(row[0], map(float, row[1:])) for row in reader]

    words, embeddings = zip(*d)
    np.save(target, np.array(embeddings))

    return {w: i for (w,i) in it.izip(words, it.count())}


def main(argv=None):
    d = store_embeddings_binary("data/"+FLAGS.dictionary_name+"/"+FLAGS.dictionary_name+".txt",
                                "data/"+FLAGS.dictionary_name+"/"+FLAGS.dictionary_name+".npy",
                                strip_not_matching=r"^[" + ALLOWED_CHARACTERS + "]+$")

    if FLAGS.train:
        text_to_ids_tfr("data/original/train.csv",
                        d,
                        "data/ids/"+FLAGS.dictionary_name+"/train_files/train",
                        "data/ids/"+FLAGS.dictionary_name+"/train_oovstat.json",
                        symmetric=True,
                        split_in_n_files=10)

    if FLAGS.test:
        text_to_ids_tfr("data/original/test.csv",
                        d,
                        "data/ids/"+FLAGS.dictionary_name+"/test_files/test",
                        "data/ids/"+FLAGS.dictionary_name+"/test_oovstat.json",
                        symmetric=False,
                        shuffle=False)

if __name__ == "__main__":
    tf.app.run()
