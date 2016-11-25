from sklearn import cluster, datasets, preprocessing
import pickle
import numpy as np
#import nltk
#import string
#from nltk.stem.lancaster import LancasterStemmer
import gensim
from sklearn import svm
#from sklearn.metrics.pairwise import chi2_kernel
import time
import re

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

feel = ["ANGER","JOY","FEAR","LOVE","SURPRISE","SADNESS"]
lable_nums = [0,1,2,3,4,5]

#exclude = set(string.punctuation)

skip = [4,3,2,1,0]

C = [0.01,0.1,1,10,100]

for ci in C:
    for inte in range(5):           ## for each of 5 cases    
        SVMM = load_obj("SVMmodel"+str(inte)+"c"+str(ci)+"kMS"+str(128))
        train_X = []
        train_y = []
        for feeli in feel:
            for feee in range(5):
                if feee !=skip[inte]:
                    pkl= open(feeli+"MSh2v"+str(feee)+".pkl","rb")
                    temp = pickle.load(pkl)
                    for t in temp[1]:
                        train_y.append(temp[0])
                        train_X.append(t)                           ## loading vecs and labels  
                    pkl.close()
                    print ("done loading vecs for " + feeli)
        print(SVMM.score(train_X,train_y))
