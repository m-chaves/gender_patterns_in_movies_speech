'''
Author: Mariana Chaves
Date: July 2022

This code does gridsearch using 3-fold on the preprocessed Cornell Dataset for logistic regression
'''

import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
# from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
# import sklearn
import pickle
# import re
# import string
# import myfunctions
# import shutil
# from tqdm import tqdm
# import matplotlib.pyplot as plt

#-------------------------------------------
# load data
#-------------------------------------------

# Load Cornell preprocessing
cornell_prepro = pd.read_pickle("datasets/cornell_corpus/cornell_prepro.pkl")  

# Lower case
cornell_prepro['text'] = cornell_prepro['text'].str.lower()

# target variable
target = (np.array(cornell_prepro['gender']) == 'F').astype(int)

#-------------------------------------------
# Work with unigrams that appear at least 5 times
#-------------------------------------------

# Using the parameter `vocabulary` from `CountVectorizer`, we can select the unigrams that we want to take into consideration. We identify the ones that appear at least 5 times and set them as our vocabulary. 

# Get vocabularies

# Get all unigrams
vectorizer = CountVectorizer(ngram_range = (1,1))
n_grams = vectorizer.fit_transform(cornell_prepro['text'])
# Identify unigrams that showed up 5 or more times
freqs = zip(vectorizer.get_feature_names_out(), np.asarray(n_grams.sum(axis=0)).ravel())
important_unigrams = [f[0] for f in freqs if f[1]>=5]
print('unigrams that show up 5 or more times:', len(important_unigrams))

# Get all unigrams, bigrams and trigrams
vectorizer = CountVectorizer(ngram_range = (1,3))
n_grams = vectorizer.fit_transform(cornell_prepro['text'])
# Identify n-grams that showed up 5 or more times
freqs = zip(vectorizer.get_feature_names_out(), np.asarray(n_grams.sum(axis=0)).ravel())
important_unibitrigrams = [f[0] for f in freqs if f[1]>=5]
print('unigrams, bigrams and trigrams that show up 5 or more times:', len(important_unibitrigrams))

#-------------------------------------------
# BOW and TF-IDF
#-------------------------------------------

# Get BOW

print('-'*50)
print('Getting bag of words matrices')
print('-'*50)

# For unigrams
BOW_model_unigrams = CountVectorizer(ngram_range = (1,1), vocabulary = important_unigrams)
BOW_unigrams = BOW_model_unigrams.fit_transform(cornell_prepro['text'])

# For unigrams, bigrams, and trigrams
BOW_model_unibitrigrams = CountVectorizer(ngram_range = (1,3), vocabulary = important_unibitrigrams)
BOW_unibitrigrams = BOW_model_unibitrigrams.fit_transform(cornell_prepro['text'])

print('-'*50)
print('Getting TF-IDF matrices')
print('-'*50)

# For unigrams
TFIDF_model_unigrams = TfidfVectorizer(ngram_range = (1,1), vocabulary = important_unigrams)
TFIDF_unigrams = TFIDF_model_unigrams.fit_transform(cornell_prepro['text'])

# For unigrams, bigrams, and trigrams
TFIDF_model_unibitrigrams = TfidfVectorizer(ngram_range = (1,3), vocabulary = important_unibitrigrams)
TFIDF_unibitrigrams = TFIDF_model_unibitrigrams.fit_transform(cornell_prepro['text'])

#-------------------------------------------
# Gridsearch on logistic regression
#-------------------------------------------

# Grid of parameters to try
C = [0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]
hyperparameters=[{'penalty':['l1'], 
                  'C': C,
                'solver':['saga','liblinear']},
                 {'penalty':['l2'],
                 'C': C,
                 'solver':['newton-cg','lbfgs','sag','liblinear','saga']},
                {'penalty':['none'],
                 'solver':['newton-cg','lbfgs','saga','sag']}]


# Using TFIDF on unigrams
print('-'*50)
print('Logistic regression using TFIDF on unigrams')
print('-'*50)

features = preprocessing.MaxAbsScaler().fit_transform(TFIDF_unigrams)
logistic_reg = LogisticRegression(max_iter = 1000)
logisticGridsearch_TFIDF_unigrams = GridSearchCV(logistic_reg, hyperparameters, cv = KFold(3), verbose = 3)
logisticGridsearch_TFIDF_unigrams.fit(features, target)

print('Best hyperameters', logisticGridsearch_TFIDF_unigrams.best_params_)
print('Best test accuracy', logisticGridsearch_TFIDF_unigrams.best_score_)

#Save gridsearch results
pickle.dump(logisticGridsearch_TFIDF_unigrams, open( "datasets/cornell_corpus/results/logisticGridsearch_TFIDF_unigrams.pickle", "wb" ))


# Using TFIDF on unigrams, bigrams and trigrams
print('-'*50)
print('Logistic regression using TFIDF on unigrams, bigrams and trigrams')
print('-'*50)

features = preprocessing.MaxAbsScaler().fit_transform(TFIDF_unibitrigrams)
logistic_reg = LogisticRegression(max_iter = 1000)
logisticGridsearch_TFIDF_unibitrigrams = GridSearchCV(logistic_reg, hyperparameters, cv = KFold(3), verbose = 1)
logisticGridsearch_TFIDF_unibitrigrams.fit(features, target)

print('Best hyperameters', logisticGridsearch_TFIDF_unibitrigrams.best_params_)
print('Best test accuracy', logisticGridsearch_TFIDF_unibitrigrams.best_score_)

#Save gridsearch results
pickle.dump(logisticGridsearch_TFIDF_unibitrigrams, open( "datasets/cornell_corpus/results/logisticGridsearch_TFIDF_unibitrigrams.pickle", "wb" ))

# Using BOW on unigrams
print('-'*50)
print('Logistic regression using BOW on unigrams')
print('-'*50)

features = preprocessing.MaxAbsScaler().fit_transform(BOW_unigrams)
logistic_reg = LogisticRegression(max_iter = 1000)
logisticGridsearch_BOW_unigrams = GridSearchCV(logistic_reg, hyperparameters, cv = KFold(3), verbose = 1)
logisticGridsearch_BOW_unigrams.fit(features, target)

print('Best hyperameters', logisticGridsearch_BOW_unigrams.best_params_)
print('Best test accuracy', logisticGridsearch_BOW_unigrams.best_score_)

#Save gridsearch results
pickle.dump(logisticGridsearch_BOW_unigrams, open( "datasets/cornell_corpus/results/logisticGridsearch_BOW_unigrams.pickle", "wb" ))


# Using BOW on unigrams, bigrams and trigrams
print('-'*50)
print('Logistic regression using BOW on unigrams, bigrams and trigrams')
print('-'*50)

# Using BOW on unigrams, bigrams and trigrams
features = preprocessing.MaxAbsScaler().fit_transform(BOW_unibitrigrams)
logistic_reg = LogisticRegression(max_iter = 1000)
logisticGridsearch_BOW_unibitrigrams = GridSearchCV(logistic_reg, hyperparameters, cv = KFold(3), verbose = 1)
logisticGridsearch_BOW_unibitrigrams.fit(features, target)

print('Best hyperameters', logisticGridsearch_BOW_unibitrigrams.best_params_)
print('Best test accuracy', logisticGridsearch_BOW_unibitrigrams.best_score_)

#Save gridsearch results
pickle.dump(logisticGridsearch_BOW_unibitrigrams, open( "datasets/cornell_corpus/results/logisticGridsearch_BOW_unibitrigrams.pickle", "wb" ))
