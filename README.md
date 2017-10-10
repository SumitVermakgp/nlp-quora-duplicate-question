## Quora Question Pairs - A deep learning experiment

The output of a project for the university: a Tensorflow implementation that faced the task posed by the Quora Question Pairs dataset, pubblicated along with the Kaggle competion. The goal is to build a classifier ables to classify whether two questions, written in english, have the same meaning. Our approach relies on pre-trained word vectors, specifically GloVe2, and deep neural network models. Two architecture have been developed and assessed side-by- side, the first in the flavour of a sentence encoder, the second more inspired by neural reasoner systems.

### Script
All the models are encoded in `main.py`, check on top of the file for the supported flags.
### Data
Use `prepare_data.py` to preprocess the data and prepare them for the training phase.
The script expects to find the original files in the `data/original` folder, you can download them from [Kaggle](https://www.kaggle.com/c/quora-question-pairs/data)
Check the supported flags on top of the python file.

### Doc
Check the [project report](report.pdf) for more details. 
