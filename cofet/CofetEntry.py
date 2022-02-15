import pandas as pd
import numpy as np
import time
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from nltk import tokenize
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from cofet.DataTransformer import DataTransformer
from cofet.ModelSelector import ModelSelector
from cofet.Evaluator import Evaluator

class CofetEntry(object):
    def __init__(self, config):
        self.config = config

    def load(self):
        self.Corpus = pd.read_csv(self.config['file'], encoding='latin-1') # CSV file containing posts

    def adapt(self):
        # preprocessing texts: stopwords, lemmatize. 
        self.Corpus = DataTransformer.populateFinalText(self, self.Corpus)

    def compose(self):
        if self.config['useTfidf'] == True: 
            self.Corpus = DataTransformer.populateTfIdf(self.Corpus)
        if self.config['useUniGram'] == True: 
            self.Corpus = DataTransformer.populateGram(self.Corpus, 1)
        if self.config['useBiGram'] == True: 
            self.Corpus = DataTransformer.populateGram(self.Corpus, 2)
        
        # remove text field after textual feature computation
        self.Corpus.drop('text_final', inplace=True, axis=1)


    def preTrain(self):
        Y = self.Corpus[self.config['labelName']]

        # features
        self.Corpus.drop('Text', inplace=True, axis=1)
        self.Corpus.drop(self.config['labelName'], inplace=True, axis=1)
        
        X = self.Corpus.replace(np.nan, 0)
        X = preprocessing.MinMaxScaler().fit_transform(X)

        self.Train_X, self.Test_X, self.Train_Y, self.Test_Y = model_selection.train_test_split(X, Y, test_size=self.config['testSize'], random_state=11, stratify=Y)
        self.Train_Y = self.Train_Y.astype(int)
        self.Test_Y = self.Test_Y.astype(int)

    def clf(self):
        self.preTrain()
        if self.config['clf'] == 'nb_clf': 
            ModelSelector.nb_clf(self)
        if self.config['clf'] == 'svm_clf': 
            ModelSelector.svm_clf(self)
        if self.config['clf'] == 'rf_clf': 
            ModelSelector.rf_clf(self)
        if self.config['clf'] == 'lr_clf': 
            ModelSelector.lr_clf(self)

    def elv(self):
        Evaluator.evl(self)