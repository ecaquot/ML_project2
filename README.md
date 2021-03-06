# ML_project2
This project was realized during the class of Machine learning in the semester of Autumn 2019 at EPFL. It aims at classifying tweets that used to contain a positive :) smiley or a negative :( smiley.

## Members
* [**Jeanne CHAVEROT**](jeanne.chaverot@epfl.ch)
* [**Etienne CAQUOT**](etienne.caquot@epfl.ch)
* [**Aslam CADER**](aslam.cader@epfl.ch)

## Code
The code is situated in the __project_text_classification__ folder, there are 3 main python files (.py) :
- `run.py` The main file that contains our best model and will produce the best predictions we could get.
- `models.py` The implementation of the other models that were not as efficient as the one in `run.py'
- `clean.py` Function we defined to help us clean the data during this project

The folder __bert__ contains [source code](https://github.com/google-research/bert) to train bert model.

The code relies on the following libraries: **pandas**, **numpy**, **nltk**, **keras**, **sklearn**, **gensim**, **h5py**, **torch**, **transformers**, **tensorflow 1.13.0rc1** You can install them easily with `pip`.

### Other models
We also have 5 files used in one of our models (for GLoVe embedding), they are not part of the best predictions but we upload them for completeness. Those files are :
- `build_vocab.sh`
- `cut_vocab.sh`
- `pickle_vocab.py`
- `cooc.py`
- `glove_solution.py`

The files should be run in this order, it is explained in `models.py`.

## Data
The data can be found on [Aicrowd](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files).
Download the data, create a folder name __Datasets__ in __project_text_classification__ and place in it the folder __twitter-datasets__ obtained from unziping __twitter-datasets.zip__.

You should have the following structure: __project_text_classification/Datasets/twitter-datasets/__, containing six files :
- `sample_submission.csv`
- `test_data.txt` The data used to make our predictions on Aicrowd.
- `train_neg_full.txt` & `train_neg.txt` The tweets that used to contain a negative :( smiley. The second file is a subset of the first one.
- `train_pos_full.txt` & `train_pos.txt` The tweets that used to contain a positive :) smiley. The second file is a subset of the first one.

We use `train_neg_full.txt`, `train_pos_full.txt` and `test_data.txt` in our code.

## Run
Here are the instructions to reproduce our best predictions.

### Prerequisite
You should download the pre-trained model available [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip). Unzip it and put the folder in __project_text_classification/bert/__ without changing the folder name (__uncased_L-12_H-768_A-12__).

### Execute
Your current directory should be __project_text_classification/__ and you can execute `python run.py`. This will create a file `submission.csv` which is our best predictions for each entry of the test set. It is composed of a series of 1 and -1, where 1 means that we predict a positive smiley :) and -1 predict negative smiley :(.

**DISCLAIMER: might take several hours even with an Nvidia P100**

## Report
You can find a detailed explanation of our work in **report.pdf**
