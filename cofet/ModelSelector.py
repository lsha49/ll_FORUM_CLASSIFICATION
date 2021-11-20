import pandas as pd
import numpy as np
import time
import gensim
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec 
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from nltk import tokenize
import math
import textstat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

class ModelSelector(object):
    def svm_clf_grid(self):
        SVM = svm.SVC()
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]} 
        parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]
        # print(SVM.get_params().keys())
        # exit()
        grid = GridSearchCV(SVM,parameters,refit=True,verbose=2)
        grid.fit(self.Train_X,self.Train_Y)
        print(grid.best_estimator_)
        exit()

        # A sample GridSearched model: 
        # SVM = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
        # decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
        # max_iter=-1, probability=False, random_state=None, shrinking=True,
        # tol=0.001, verbose=False)
    
    def rf_clf_grid(self):
        rfc=RandomForestClassifier()
        parameters = {
            'n_estimators': [50, 150, 250],
            'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
            'min_samples_split': [2, 4, 6]
        }
        print(rfc.get_params().keys())
        grid = GridSearchCV(rfc,parameters,refit=True,verbose=2)
        grid.fit(self.Train_X,self.Train_Y)
        # print(grid.best_estimator_)
        # exit()

        # A sample GridSearched model: 
        # rfc=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #     max_depth=None, max_features=0.5, max_leaf_nodes=None,
        #     min_impurity_split=1e-07, min_samples_leaf=1,
        #     min_samples_split=4, min_weight_fraction_leaf=0.0,
        #     n_estimators=250, n_jobs=1, oob_score=False, random_state=None,
        #     verbose=0, warm_start=False)
    
    def lr_clf_grid(self):
        lrc=LogisticRegression()
        parameters = {
            'penalty' : ['l1', 'l2'],
            'C' : np.logspace(-4, 4, 20),
            'solver' : ['liblinear']
        }
        print(lrc.get_params().keys())
        grid = GridSearchCV(lrc,parameters,refit=True,verbose=2)
        grid.fit(self.Train_X,self.Train_Y)
        # print(grid.best_estimator_)
        # exit()

        # lrc = LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,
        #   fit_intercept=True, intercept_scaling=1, max_iter=100,
        #   multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
        #   solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

    def nb_clf(self):
        GaussianNaive = naive_bayes.GaussianNB()
        GaussianNaive.fit(self.Train_X,self.Train_Y)
        Gpredictions_NB = GaussianNaive.predict(self.Test_X)

        self.predicted = GaussianNaive.predict(self.Test_X)
        self.predicted_prob = GaussianNaive.predict_proba(self.Test_X)

    def svm_clf(self):
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(self.Train_X, self.Train_Y)      

        self.predicted = SVM.predict(self.Test_X)
        self.predicted_prob = SVM.predict_proba(self.Test_X)


    def rf_clf(self):
        rfc=RandomForestClassifier(n_estimators=100)
        rfc.fit(self.Train_X,self.Train_Y)
        self.predicted = rfc.predict(self.Test_X)
        self.predicted_prob = rfc.predict_proba(self.Test_X)

    def lr_clf(self):
        lrc=LogisticRegression(random_state=0)
        lrc.fit(self.Train_X,self.Train_Y)
        self.predicted = lrc.predict(self.Test_X)
        self.predicted_prob = lrc.predict_proba(self.Test_X)
