from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import logging

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from gensim.models import word2vec

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
from keras_preprocessing import text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

''' ---------- Logistic Regression and GloVe ---------- 
First run the scripts
 - build_vocab.sh
 - cut_vocab.sh
 - python3 pickle_vocab.py
 - python3 cooc.py '''

def logistic_regression_glove(positive_tweets,negative_tweets):
    """ Return a logistic Regression model fitted on GloVe embeddings

    Keyword arguments:
    positive_tweets -- the file (.txt) that contains the positive tweets
    negative_tweets -- the file (.txt) that contains the negative tweets
    """
    emb = np.load('embeddings.npy')

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    #embedd positive tweets 
    num_lines_pos = sum(1 for line in open(positive_tweets))

    train_pos = np.zeros((num_lines_pos,emb.shape[1]))
    with open(positive_tweets) as f:
        for line_index, line in enumerate(f):
            words = line.split()
            index = [vocab[word] for word in words if word in vocab.keys()]
            line_fet = np.mean(np.array([emb[i] for i in index]),axis = 0)
            train_pos[line_index] = line_fet

    index_to_remove_pos = np.unique([x for x,y in np.argwhere(np.isnan(train_pos))])

    train_pos_2 = np.delete(train_pos,index_to_remove_pos,axis = 0)


    #embedd negative tweets 
    num_lines_neg = sum(1 for line in open(negative_tweets))

    train_neg = np.zeros((num_lines_neg,emb.shape[1]))
    with open(negative_tweets) as f:
        for line_index, line in enumerate(f):
            words = line.split()
            index = [vocab[word] for word in words if word in vocab.keys()]
            line_fet = np.mean(np.array([emb[i] for i in index]),axis = 0)
            train_neg[line_index] = line_fet

    index_to_remove_neg = np.unique([x for x,y in np.argwhere(np.isnan(train_neg))])

    train_neg_2 = np.delete(train_neg,index_to_remove_neg,axis = 0)

    # Combine positive and negative tweets to have the whole training set
    X = np.vstack((train_pos_2,train_neg_2))
    y_pos = np.ones(train_pos_2.shape[0])
    y_neg = np.repeat(-1,train_neg_2.shape[0])
    Y = np.hstack((y_pos,y_neg))

    #Train a Logistic Regression classifier
    logiCV = LogisticRegressionCV(Cs=5, fit_intercept=True, cv=4, dual=False, penalty='l2', scoring=None,
                        solver='sag', tol=0.0001, max_iter=10000, class_weight=None, n_jobs=-1, verbose=0,
                        refit=True, intercept_scaling=1.0, multi_class='ovr', random_state=None, l1_ratios=None)

    logiCV.fit(X,Y)

    return logiCV


''' ---------- Logistic Regression and Word2vec ----------'''

def logistic_regression_word2vec(positive_tweets,negative_tweets):
    """ Return a logistic Regression model fitted on Word2vec embeddings

    Keyword arguments:
    positive_tweets -- the file (.txt) that contains the positive tweets
    negative_tweets -- the file (.txt) that contains the negative tweets
    """
    f = open(positive_tweets)
    tweets_pos = [line.split() for line in f.readlines()]
    f.close()

    f = open(negative_tweets)
    tweets_neg = [line.split() for line in f.readlines()]
    f.close()

    # Parameters for Word2vec
    size = 300
    min_count = 5
    epoch = 10

    #training Word2vec on tweets
    model = word2vec.Word2Vec(sentences=tweets_pos + tweets_neg, corpus_file=None, size=size, alpha=0.025, window=5,
                          min_count=min_count, max_vocab_size=None, sample=0.001, seed=1, workers=1, min_alpha=0.0001, sg=0,
                          hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, iter=epoch, null_word=0, trim_rule=None,
                          sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), max_final_vocab=None)

    #embedd positive tweets 
    train_pos = np.zeros((len(tweets_pos),size))
    for index, tokens in enumerate(tweets_pos):
        vect = [model.wv[token] for token in tokens if token in model.wv]
        train_pos[index] = np.mean(vect, axis = 0)

    index_to_remove_pos = np.unique([x for x,y in np.argwhere(np.isnan(train_pos))])

    train_pos_2 = np.delete(train_pos,index_to_remove_pos,axis = 0)

    #embedd negative tweets 
    train_neg = np.zeros((len(tweets_neg),size))
    for index, tokens in enumerate(tweets_neg):
        vect = [model.wv[token] for token in tokens if token in model.wv]
        train_neg[index] = np.mean(vect, axis = 0)

    index_to_remove_neg = np.unique([x for x,y in np.argwhere(np.isnan(train_neg))])

    train_neg_2 = np.delete(train_neg,index_to_remove_neg,axis = 0)

    
    # Combine positive and negative tweets to have the whole training set
    X = np.vstack((train_pos_2,train_neg_2))
    y_pos = np.ones(train_pos_2.shape[0])
    y_neg = np.repeat(-1,train_neg_2.shape[0])
    Y = np.hstack((y_pos,y_neg))

    #Train a Logistic Regression classifier
    logiCV = LogisticRegressionCV(Cs=5, fit_intercept=True, cv=4, dual=False, penalty='l2', scoring=None,
                        solver='sag', tol=0.0001, max_iter=10000, class_weight=None, n_jobs=-1, verbose=0,
                        refit=True, intercept_scaling=1.0, multi_class='ovr', random_state=None, l1_ratios=None)

    logiCV.fit(X,Y)

    return logiCV


''' ---------- Vader (NLTK) ----------'''

def Vader(test_tweets):
    """ Return prediction for test_tweets

    Keyword arguments:
    test_tweets -- the file (.txt) that contains the test tweets
    """
    f = open("Datasets/twitter-datasets/test_data.txt")
    tweets = [line for line in f.readlines()]
    f.close()

    prediction = []
    stop_words = set(stopwords.words('english'))
    sid = SentimentIntensityAnalyzer()
    for tweet in tweets:
        tokens = word_tokenize(tweet)
        result = [i for i in tokens if not i in stop_words]
        result = ' '.join(result)

        ss = sid.polarity_scores(tweet)
        if ss['neu'] == 1:
            prediction.append(-1)
        elif ss['neg'] > ss['pos']:
            prediction.append(-1)
        else:
            prediction.append(1)

    return prediction


''' ---------- LSTM ----------'''

def LSTM(positive_tweets,negative_tweets):
    """ Return a LSTM model fitted on postive_tweets and negative_tweets

    Keyword arguments:
    positive_tweets -- the file (.csv) that contains the positive tweets
    negative_tweets -- the file (.csv) that contains the negative tweets
    """
    pos_df = pd.read_csv(positive_tweets, index_col=0)
    neg_df = pd.read_csv(negative_tweets, index_col=0)

    train = pd.concat([pos_df,neg_df])

    #Randomize order
    train = train.sample(frac=1, random_state = 1)

    train = train.dropna()

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 450000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 300
    tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train.tweets.values)
    word_index = tokenizer.word_index

    #number of word with that appears at least 5 times
    words = len([k for k in tokenizer.word_index.keys() if tokenizer.word_counts[k] > 4])

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = words
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 300
    tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train.tweets.values)
    word_index = tokenizer.word_index

    #Create sequence of index
    X = tokenizer.texts_to_sequences(train.tweets.values)
    X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    Y = train.label

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 1)

    #Build model
    batch_size = 8192

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    #To have our best model
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    #Training
    model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=250,
            validation_data=(X_test, Y_test), callbacks=[es, mc])

    saved_model = load_model('best_model.h5')

    return saved_model
