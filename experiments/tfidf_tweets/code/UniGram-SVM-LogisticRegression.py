# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import pickle
import chardet
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

writer = csv.writer(open("UniGram-SVC-LogisticRegression-Filters.csv","a"))

writer.writerow(["Percent of Train  Data", "Classifier Specs" ,"Train Docs(Count)" , "Test Docs(Count)"  , "Train Error" ,"Test Error"])

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

#global dicts
datatr = dict()
datate =dict()

def read_datasetstr(flist, t_type,num):
    # for duplicate removal
    tot = int((len(flist)*num)/100)
    for l in flist[:tot+1]:
        datatr[l] = t_type

def read_datasetste(flist, t_type):
    # for duplicate removal
    for l in flist:
        datate[l] = t_type



# for n in [0.1,0.5,1,10,20,50]:
for n in [100]:
    print("With "+str(n)+" Percent of Training Data !!")
    print("="*80)
    print()
    print()
    # ##Pre-Preocessed
    # # read in joy , disgust, sadness, shame, anger, guilt, fear training dataset
    # read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterjoyPTr"), 'joy',n)
    # read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterangerPTr"), 'anger',n)
    # read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MastersadnessPTr"), 'sadness',n)
    # read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MastersurprisePTr"), 'surprise',n)
    # read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterlovePTr"), 'love',n)
    # read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterfearPTr"), 'fear',n)

    # # read in joy , disgust, sadness, shame, anger, guilt, fear test dataset
    # read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterjoyPTe"), 'joy')
    # read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterangerPTe"), 'anger')
    # read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MastersadnessPTe"), 'sadness')
    # read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MastersurprisePTe"), 'surprise')
    # read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterlovePTe"), 'love')
    # read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterfearPTe"), 'fear')

    # DesignMatrix =  [ "#%~-".join([k.decode("utf-8"),v]) for k, v in datatr.items() ]
    # TestMatrix = [ "#%~-".join([k.decode("utf-8"),v]) for k, v in datate.items() ]


    # Unpreprocessed Data
    # read in joy , disgust, sadness, shame, anger, guilt, fear training dataset
    read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/Tr/MasterjoyTr"), 'joy',n)
    read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/Tr/MasterangerTr"), 'anger',n)
    read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/Tr/MastersadnessTr"), 'sadness',n)
    read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/Tr/MastersurpriseTr"), 'surprise',n)
    read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/Tr/MasterloveTr"), 'love',n)
    read_datasetstr(load_obj("DATA/Preproccess Tools/Preprocessor/Tr/MasterfearTr"), 'fear',n)


    # read in joy , disgust, sadness, shame, anger, guilt, fear test dataset
    read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/Te/MasterjoyTest"), 'joy')
    read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/Te/MasterangerTest"), 'anger')
    read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/Te/MastersadnessTest"), 'sadness')
    read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/Te/MastersurpriseTest"), 'surprise')
    read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/Te/MasterloveTest"), 'love')
    read_datasetste(load_obj("DATA/Preproccess Tools/Preprocessor/Te/MasterfearTest"), 'fear')

    DesignMatrix =  [ "#%~-".join([k,v]) for k, v in datatr.items() ]
    TestMatrix = [ "#%~-".join([k,v]) for k, v in datate.items() ]

    common = set(DesignMatrix).intersection(set(TestMatrix))

    DesignMatrix = set(DesignMatrix) - common
    TestMatrix = set(TestMatrix) - common

    DesignMatrix = [[k.split("#%~-")[0].encode("utf-8"),k.split("#%~-")[1]] for k in DesignMatrix]
    TestMatrix = [[k.split("#%~-")[0].encode("utf-8"),k.split("#%~-")[1]] for k in TestMatrix]


    X_test=[]
    X_train =[]
    y_test=[]
    y_train=[]

    for i in DesignMatrix:
        X_train.append(i[0])
        y_train.append(i[1])

    for i in TestMatrix:
        X_test.append(i[0])
        y_test.append(i[1])

    # print(X[0])
    print(len(X_train))

    #decoding
    # XD=[]
    # for x1 in X:
    #     #print(x1)
    #     try:
    #         XD.append(x1.decode(chardet.detect(x1)['encoding']))
    #     except:
    #         XD.append(x1)
    # print(len(XD))
    # XD=[]
    # for x in X:
    #     XD.append(x.decode("utf-8"))
    # print(len(XD))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print('Data loaded.....')

    categories = ["anger","fear","joy","love","sadness","surprise"]

    #print(X_train[1])
    def size_mb(docs):
        return sum(len(s) for s in docs) / 1e6

    data_train_size_mb = size_mb(X_train)
    data_test_size_mb = size_mb(X_test)

    print("%d documents - %0.3fMB (training set)" % (
        len(X_train), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(X_test), data_test_size_mb))
    print("%d categories" % len(categories))
    print()

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()

    # vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words=None)
    vectorizer = CountVectorizer(min_df=1)
    X_train = vectorizer.fit_transform(X_train)
    duration = time() - t0

    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()

    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    # mapping from integer feature name to original token string
    feature_names = vectorizer.get_feature_names()
    # ### Vary K Value
    # if X_test.shape[1] >= 3500:
    #     kkk=3500
    # else:
    #     kkk="all"
    # print("Extracting "+str(kkk) +" best features by a chi-squared test")
    # t0 = time()
    # ch2 = SelectKBest(chi2, k=kkk)
    # X_train = ch2.fit_transform(X_train, y_train)
    # X_test = ch2.transform(X_test)

    # feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    # print("done in %fs" % (time() - t0))
    # print()
    feature_names = np.asarray(feature_names)


    def trim(s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."


    ###############################################################################
    # Benchmark classifiers
    # Bench Mark Result Print Function
    def benchmark(clf):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        print("Training Error:")
        t0 = time()
        pred = clf.predict(X_train)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        Trscore = metrics.accuracy_score(y_train, pred)
        print("accuracy:   %0.3f" % Trscore)
        print("error: " + str(1-Trscore))

        print("Testing Error:")
        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        Tescore = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % Tescore)
        print("error: " + str(1-Tescore))

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (category, " ".join(feature_names[top10]).encode("utf-8"))))
            print()

        print("classification report:")
        print(metrics.classification_report(y_test, pred,target_names=categories))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        writer.writerow([str(n)+" %", clf_descr ,str(X_train.shape[0]), str(X_test.shape[0])  , str((1-Trscore)*100) +" %" ,str((1-Tescore)*100) +" %"])
        return clf_descr, Tescore, train_time, test_time

    results = []

    # for penalty in ["l2", "l1"]:
    print('=' * 80)
    # print("%s penalty" % penalty.upper())
    print("L2 penalty")
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty="l2",dual=False, tol=1e-3)))
    results.append(benchmark(LogisticRegression()))
