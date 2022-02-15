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
from sklearn.feature_extraction.text import TfidfVectorizer
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
import math
import textstat
from sklearn.feature_extraction.text import CountVectorizer

class Evaluator(object):

    def evl(self):
        predicted = self.predicted 
        predicted_prob = self.predicted_prob #for multi class AUC
        print('model result of ', self.config['clf'])
        print("Accuracy Score -> ",accuracy_score(predicted, self.Test_Y))
        print("Kappa Score -> ",cohen_kappa_score(predicted, self.Test_Y))
        print("ROC AUC Score -> ",roc_auc_score(self.Test_Y, predicted))
        print("F1 Score -> ",f1_score(predicted, self.Test_Y))

        # for multi class
        # print("ROC AUC Score -> ", roc_auc_score(self.Test_Y, predicted_prob, average='weighted', multi_class='ovo'))
        # print("F1 Score -> ",f1_score(predicted, self.Test_Y, average='weighted'))
