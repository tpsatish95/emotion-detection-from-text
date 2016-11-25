#pickle has protocol error
### py 3 has a higher protocol py 2 has
###pickle.dump(your_object, your_file, protocol=2)

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
import tokenize

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


kg = [128,190,256]
for ki in kg:    ### testing part to get optimal C
    for ci in C:
        avgALL5 = []
        for inte in range(5):           ## for each of 5 cases
            SVMM = load_obj("SVMmodel"+str(inte)+"c"+str(ci)+"kMS"+str(ki))
            count = [0,0,0,0,0,0]
            scount = [0,0,0,0,0,0]
            fi = 0
            for feeli in feel:
                skk = open(feeli+"MSh2v"+str(skip[inte])+".pkl","rb")           
                while 1:
                        try:
                            temp = pickle.load(skk)
                            count[fi]=len(temp[1])
                            for t in temp[1]:
                                #train_y.append(temp[0])
                                index = SVMM.predict(t)
                                if index[0] == temp[0]:
                                    scount[fi]=scount[fi]+1
                            skk.close()
                        except:
                            skk.close()
                            print ("done loading vecs for " +feeli+"  "+str((scount[fi]/count[fi])*100))
                            break
                fi=fi+1
            flo = []
            for feei in range(6):
                flo.append((scount[feei]/count[feei])*100)
            avgALL5.append(flo)
        avgg = []
        for eemo in range(6):
            t=0
            for ra in range(5):
                t=t+avgALL5[ra][eemo]
            avgg.append(t/5)
        store = open("RMS"+str(ki)+"C"+str(ci)+".txt","w")
        for eemo in range(6):
            store.write("EMO "+str(feel[eemo])+" Accu "+str(avgg[eemo])+"\n")
        store.close()