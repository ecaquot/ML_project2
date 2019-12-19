
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
import time

import csv

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})



path_bert = './bert/'
data_bert_path =  path_bert + 'data/'
model_path = path_bert + 'uncased_L-12_H-768_A-12/'

raw_data_path = './Datasets/twitter-datasets/'

train_pos_path = raw_data_path + 'train_pos_full.txt'
train_neg_path = raw_data_path + 'train_neg_full.txt'
test_data_path = raw_data_path + 'test_data.txt'



# Import data
print('Import data...')
# 0 for bert, -1 for our prediction
negative_tag = 0

train_pos_df = pd.read_csv(train_pos_path, delimiter="\\n", header=None, names = ["tweets"], engine='python')
train_pos_df["prediction"] = 1
train_neg_df = pd.read_csv(train_neg_path, delimiter="\\n", header=None, names = ["tweets"], engine='python')
train_neg_df["prediction"] = negative_tag

# concatenate train positive and negative
to_concat = [train_pos_df, train_neg_df]
df_train = pd.concat(to_concat)

# Split randomly the train set into train/set (80%-20%) in order to evaluate performance:
msk = np.random.rand(len(df_train)) < 0.8

train = df_train[msk]
test = df_train[~msk]


# print to see if it has been shuffled correctly
positive_in_train = len(train[train["prediction"] == 1])
negative_in_train = len(train[train["prediction"] == negative_tag])

positive_in_test =len(test[test["prediction"] == 1])
negative_in_test = len(test[test["prediction"] == negative_tag])

print("There is {} positive, {} negative tweets in train set ".format(positive_in_train, negative_in_train))

print("There is {} positive, {} negative tweets in test set ".format(positive_in_test, negative_in_test))


print('Prepare data for BERT...')
# Prepare for BERT the dataframes:
train_df_bert = pd.DataFrame({
    'id':train.index,
    'label':train["prediction"],
    'alpha':['a']*train.shape[0],
    'text': train["tweets"].replace(r'\n', ' ', regex=True)
})

dev_df_bert = pd.DataFrame({
    'id':test.index,
    'label':test["prediction"],
    'alpha':['a']*test.shape[0],
    'text': test["tweets"].replace(r'\n', ' ', regex=True)
})



# for prediction
df_test = pd.read_csv(test_data_path, sep='^([^,]+),', engine='python', header=None, names = ["index", "tweets"]).reset_index(drop=True)

test_df_bert = pd.DataFrame({
    'id':df_test["index"],
    'text': df_test["tweets"].replace(r'\n', ' ', regex=True)
})



# export
print('Export data in {}...'.format(data_bert_path))
train_df_bert.sample(frac=1).to_csv(data_bert_path + 'train.tsv', sep='\t', index=False, header=False)
dev_df_bert.sample(frac=1).to_csv(data_bert_path + 'dev.tsv', sep='\t', index=False, header=False)
test_df_bert.to_csv(data_bert_path + 'test.tsv', sep='\t', index=False, header=True)



print('Data creation done...')

path_classifier = path_bert + 'run_classifier.py'
# data_dir = data_bert_path
vocab_file_dir = model_path + 'vocab.txt'
config_file_dir = model_path + 'bert_config.json'
init_checkpoint = model_path + 'bert_model.ckpt'
output_dir = path_bert + 'bert_output/'

# parameters
max_seq_length=128
train_batch_size=1
learning_rate=2e-5
num_train_epochs=1.0
bashCommand = '''python {path_classifier} --task_name=cola --do_train=true --do_eval=true --do_predict=true --data_dir={data_bert_path} --vocab_file={vocab_file_dir} --bert_config_file={config_file_dir} --init_checkpoint={init_checkpoint} --max_seq_length={max_seq_length} --train_batch_size={train_batch_size} --learning_rate={learning_rate} --num_train_epochs={num_train_epochs} --output_dir={output_dir} --do_lower_case=True'''.format(
                    path_classifier = path_classifier,
                    data_bert_path = data_bert_path,
                    vocab_file_dir = vocab_file_dir,
                    config_file_dir = config_file_dir,
                    init_checkpoint = init_checkpoint,
                    max_seq_length =  max_seq_length,
                    train_batch_size = train_batch_size,
                    learning_rate = learning_rate,
                    num_train_epochs = num_train_epochs,
                    output_dir = output_dir
                 )

print("Model training and prediction begins...")
os.system(bashCommand)
print("Model training and prediction ends... ")

prediction_path = output_dir + 'test_results.tsv'
prediction =  pd.read_csv(prediction_path, delimiter='\t', header=None)
prediction.columns = [-1, 1]
prediction['Prediction'] = prediction[-1].apply(lambda x: -1 if x>0.5 else 1)

print('creating submission...')
create_csv_submission(range(1,10001), prediction['Prediction'], './submission.csv')
