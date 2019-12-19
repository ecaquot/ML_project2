import pandas as pd
import numpy as np
import re
import string

from collections import Counter
from nltk.corpus import stopwords

from keras_preprocessing import text

from itertools import groupby

#load datasets
pos_df = pd.read_csv("Datasets/twitter-datasets/train_pos_full.txt", delimiter="\\n", header=None,
                     names = ["tweets"], engine='python')
neg_df = pd.read_csv("Datasets/twitter-datasets/train_neg_full.txt", delimiter="\\n", header=None,
                     names = ["tweets"], engine='python')
test_df = pd.read_csv("Datasets/twitter-datasets/test_data.txt", delimiter="\\n", header=None,
                     names = ["tweets"], engine='python')
test_df = pd.DataFrame(test_df.tweets.str.split(',',1).tolist(), columns = ['id','tweets'])


#set labels
pos_df['label'] = 1
neg_df['label'] = 0

def preprocess(df):

    #remove numbers
    df.tweets = df.tweets.str.replace('\d+', '')
    #remove <user>
    df.tweets = df.tweets.str.replace("<user>", "")
    #remove <url>
    df.tweets = pos_df.tweets.str.replace("<url>", "")
    #remove punctuation
    df.tweets = df.tweets.apply(lambda tweet: "".join([char.lower() for char in tweet if char not in string.punctuation]))

    return df

#we have run this setting pct = 80 as it is the percentage providing the best accuracy
def change_most_common_terms(pos_df, neg_df, test_df, pct):

    results_pos = Counter()
    pos_df.tweets.str.lower().str.split().apply(results_pos.update)
    results_neg = Counter()
    neg_df.tweets.str.lower().str.split().apply(results_neg.update)

    count_pos = results_pos.most_common(1000)
    count_neg = results_neg.most_common(1000)

    dict_count_pos = dict(count_pos)
    dict_count_neg = dict(count_neg)

    both = dict_count_pos.keys() & (dict_count_neg.keys())

    count_total = [(word, dict_count_pos[word]+dict_count_neg[word]) for word in both]
    dict_both = dict(count_total)

    positive = [(word, dict_count_pos[word]) for word in both if 100*dict_count_pos[word]/dict_both[word]>pct]
    negative = [(word, dict_count_neg[word]) for word in both if 100*dict_count_neg[word]/dict_both[word]>pct]

    positive = [i[0] for i in positive]
    negative = [i[0] for i in negative]

    dic = {}
    for el in positive:
        dic[el] = 'happy'

    for el in negative:
        dic[el] = 'sad'

    pos_df.tweets = pos_df.tweets.apply(lambda tweet: " ".join([dic[word] if word in dic.keys() else word for word in tweet.split(" ")]))

    neg_df.tweets = neg_df.tweets.apply(lambda tweet: " ".join([dic[word] if word in dic.keys() else word for word in tweet.split(" ")]))

    test_df.tweets = test_df.tweets.apply(lambda tweet: " ".join([dic[word] if word in dic.keys() else word for word in tweet.split(" ")]))

    return pos_df, neg_df, test_df

    #we have run this using nb = 5
    def delete_least_common_terms(pos_df, neg_df, test_df, nb):
        results_pos = Counter()
        pos_df.tweets.str.lower().str.split().apply(results_pos.update)
        results_neg = Counter()
        neg_df.tweets.str.lower().str.split().apply(results_neg.update)

        results = results_pos + results_neg
        results = list(results.items())

        #keep all words that appear at most `nb` times in the training (positive + negative) set
        least_common = [t[0] for t in results if t[1] < (nb + 1)]

        pos_df.tweets = pos_df.tweets.apply(lambda tweet:" ".join(word for word in tweet.split(" ") if not word in stop_words))
        neg_df.tweets = neg_df.tweets.apply(lambda tweet:" ".join(word for word in tweet.split(" ") if not word in stop_words))
        test_df.tweets = test_df.tweets.apply(lambda tweet:" ".join(word for word in tweet.split(" ") if not word in stop_words))

        return pos_df, neg_df, test_df
