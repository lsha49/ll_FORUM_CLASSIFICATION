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

class DataTransformer(object):
    def populateTfIdf(self, Corpus):
        numOfGramFeatures = self.config['numOfGramFeatures']
        Tfidf_vect = TfidfVectorizer(max_features=numOfGramFeatures, ngram_range=(2,2))
        Tfidf_vect.fit(Corpus['text_final'])
        TfidfFeature = Tfidf_vect.transform(Corpus['text_final'])
        Tfidf_vect_uni = TfidfVectorizer(max_features=numOfGramFeatures, ngram_range=(1,1))
        Tfidf_vect_uni.fit(Corpus['text_final'])
        TfidfFeatureUni = Tfidf_vect_uni.transform(Corpus['text_final'])
        tfidfHeaders = Tfidf_vect.vocabulary_
        tfidfHeadersUni = Tfidf_vect_uni.vocabulary_
        keyedIndex = 0
        for key,entry in tfidfHeaders.items():
            Corpus['tf ' + key] = TfidfFeature.getcol(keyedIndex).toarray()
            keyedIndex = keyedIndex + 1
        keyedIndex = 0
        for key,entry in tfidfHeadersUni.items():
            Corpus['tf ' + key] = TfidfFeatureUni.getcol(keyedIndex).toarray()
            keyedIndex = keyedIndex + 1
        # Corpus.drop('text_final', inplace=True, axis=1)
        return Corpus


    def populateGram(self, Corpus, gramN):
        numOfGramFeatures = self.config['numOfGramFeatures']
        ngram_vectorizer = CountVectorizer(max_features=100, ngram_range=(gramN,gramN))
        ngFeature = ngram_vectorizer.fit_transform(Corpus['text_final'])
        ngHeaders = ngram_vectorizer.vocabulary_
        keyedIndex = 0
        for key,entry in ngHeaders.items():
            Corpus['ng ' + key] = ngFeature.getcol(keyedIndex).toarray()
            keyedIndex = keyedIndex + 1
        return Corpus

    def populateFinalText(self, Corpus): 
        Corpus['Text'].dropna(inplace=True)
        Corpus['Text'] = [ entry.lower() if isinstance(entry, str) else entry for entry in Corpus['Text'] ]
        Corpus['Text']=Corpus['Text'].replace(to_replace= r'\\', value= '', regex=True)
        Corpus['Text'] = [ entry if isinstance(entry, str) else entry for entry in Corpus['Text'] ]
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        for index,entry in enumerate(Corpus['Text']):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            if isinstance(entry, Iterable):
                wordList = tokenize.word_tokenize(entry)
                for word, tag in pos_tag(wordList):
                    if word not in stopwords.words('english') and word.isalpha():
                        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                        Final_words.append(word_Final)
            Corpus.loc[index,'text_final'] = str(Final_words)
        return Corpus
    
    def feature_readability(self, data):
        return textstat.automated_readability_index(data)

    def feature_ngram(self, data, numOfGramFeatures):
        ngram_vectorizer_uni = CountVectorizer(max_features=numOfGramFeatures, ngram_range=(1,1)) # unigram 
        return ngram_vectorizer_uni.fit_transform(data)
    
    def feature_tfidf(self, data, numOfGramFeatures):
        Tfidf_vect = TfidfVectorizer(max_features=numOfGramFeatures, ngram_range=(1,1))
        Tfidf_vect.fit(data)
        return Tfidf_vect.transform(data)

    def exam_selectKbest(self, data, X, Y, numOfGramFeatures):
        kval = 0
        if kval > 0:
            selector = SelectKBest(chi2, k=kval)
            selector.fit_transform(X, Y)
            alabels = data.columns.tolist()
            slabels = selector.get_support()
            print('Selected K features: ==============')
            for aindex in range(len(alabels)):
                if slabels[aindex] == True:
                    print(alabels[aindex], ',' ,end = '')
            print('\n')
            