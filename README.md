# ML_project2
Project 2 EPFL Machine Learning Course, Fall 2019

# Prerequesite
Your current directory should be __project_text_classification/__ and you can execute __run.py__ (**python run.py**).

You should download the pre-trained model available [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip). Unzip it and put the folder in __project_text_classification/bert/__ without changing the folder name (__uncased_L-12_H-768_A-12__).

**twitter-datasets** should be unzipped and put the 3 files in __project_text_classification/Datasets/twitter-datasets/__

In order to successfully train the model, please make sure to have the following libraries (all of them can be installed with __pip__):

* tensorflow **1.13.0rc1** (newer version won't work)
* **transformers**
* numpy
* pandas
* os
* sklearn
* torch
* warnings
* time
* csv

submission.csv will be generated in the same folder as run.py.

**DISCLAIMER: might take several hours even with a Nvidia P100**

## Members
* [**Jeanne CHAVEROT**](jeanne.chaverot@epfl.ch)
* [**Etienne CAQUOT**](etienne.caquot@epfl.ch)
* [**Aslam CADER**](aslam.cader@epfl.ch)
