#pickle has protocol error
### py 3 has a higher protocol py 2 has
###pickle.dump(your_object, your_file, protocol=2)
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn import cluster, datasets, preprocessing
import pickle
#import nltk
#import string
#from nltk.stem.lancaster import LancasterStemmer
import gensim
#from sklearn import svm
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



model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)
min_max_scaler = preprocessing.MinMaxScaler()
model.init_sims(replace=True)

#### getting all vecs from w2v using the inbuilt syn0 list see code

X = np.array(model.syn0)

### scaling feature vecs

X_Scaled_Feature_Vecs = min_max_scaler.fit_transform(X)

##### minimizing intertia or energy for a single cluster size k

##intialize k
k = [1024,1224,2048]

### max K that we can reach

###MAX_K = 8192

#### feeel
feel = ["ANGER","JOY","FEAR","LOVE","SURPRISE","SADNESS"]
lable_nums = [0,1,2,3,4,5]


##### better code

for ki in k:
    
    print("Bk===" +str(ki))
    t0 = time.time()
    kmeans = cluster.MiniBatchKMeans(n_clusters=ki,n_init=100,batch_size = ki*3).fit(X_Scaled_Feature_Vecs)
    print(str(time.time()-t0))
    print(kmeans.inertia_)
    print("Donek===" +str(ki))
    ##### save best for future use
    save_obj(kmeans,"EMSbestKMEANSforKM"+str(ki))
    KM = load_obj("EMSbestKMEANSforKM"+str(ki))
    i=0
    Cluster_lookUP = dict(zip(model.vocab, KM.labels_))
    for feeli in feel:  
        #hisvec = []
        #hisvecS = []
        #DumpP = open(feeli+"Mh2v"+str(ite)+".pkl","wb")
        #DumpPS = open(feeli+"Mh2vSkip"+str(ite)+".pkl","wb")
        #for sp in range(5):
        hisvec = []
        DumpP = open(feeli+"MSh2vM"+".pkl","wb")
        w = open("E/"+feeli+".txt","r")
        print(feeli)
        lines = w.readlines()
        w.close()
        ti = time.time()
        for line in lines:
            #tokens = []
            words = re.findall(r"[\w']+|[.,/;=?!$%]",line.lower())
            
            #words = line.split()
            #print(words)
            #Uwords = [w for w in words if w not in exclude]
            #for word in words:
            #   tokens.append(word.strip())  #stemmer.stem() removed
            #print(tokens)
            Svec = []
            for word in words:
                try:
                    # Svec.append(KM.predict(model[word.strip()])[0])
                    Svec.append(Cluster_lookUP[word.strip()])
                except:
                    ###Svec.append(-1)
                    pass  ##token not found so undefined bin we dont have training data for undefined so neglect
            histoVec, bin_edges = np.histogram(Svec, bins=range(ki+1))
            histoVec = [float(his) for his in histoVec]
            #print(histoVec)
            hisvec.append(histoVec)
        print(str(time.time()-ti))
        h2vN = preprocessing.normalize(hisvec)
        #h2vS = preprocessing.scale(h2vN)
        h2vS = min_max_scaler.fit_transform(h2vN)
            ##### the chi2_kernel accepts only positive values
            ##### scales feature vector between 0 and 1



            #print(histoVec)    #30 dimension vec of sentence
            #svmModel = load_obj("SVMmodel")
            #index = svmModel.predict(histoVec)
            #fe.write(str(index))
            
        """
            h2vS has the entire 2d array of vectors of a file (4 files) each row is a sentence  
            you will get 5 pickle files for each emotion to iterate for all C values of SVM time
        """
        pickle.dump([lable_nums[i], h2vS],DumpP, protocol=2)
        print(feeli+" and MSSkip for Test also Dumped!!")
        DumpP.close()
        i=i+1

    train_X = []
    train_y = []
    for feeli in feel:
        pkl= open(feeli+"MSh2vM"+".pkl","rb")
        temp = pickle.load(pkl)
        for t in temp[1]:
            train_y.append(temp[0])
            train_X.append(t)                           ## loading vecs and labels  
        pkl.close()
        print ("done loading vecs for " + feeli)
    t0 = time.time()
    C_range = 10.0 ** np.arange(-1, 1)
    gamma_range = [0.0]
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=train_y, n_folds=5)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(train_X, train_y)
    print(str(time.time()-t0))
    store = open("ERMSM"+str(ki)+".txt","w")
    store.write(str(grid.best_estimator_))
    store.write("\n"+str(grid.best_score_))
    store.close()
    save_obj(grid,str(ki)+"GridSM")
    print("Done")

#exclude = set(string.punctuation)

#skip = [4,3,2,1,0]

''''
inertiaP = 0
inertiaC = 0


for i in iss:
    print(str(i) + "start")
    kmeans = cluster.KMeans(n_clusters=k,n_init=100).fit(X_Scaled_Feature_Vecs)
    inertiaC = kmeans.inertia_
    if i==0:
        inertiaP = inertiaC
        save_obj(kmeans,"bestIforK"+str(k))
    elif inertiaC < inertiaP:
        inertiaP = inertiaC
        save_obj(kmeans,"bestIforK"+str(k))
    else:
        continue
    print(i)

    n_init : int, default: 10
Number of time the k-means algorithm will be run with different centroid seeds. 
The final results will be the best output of n_init consecutive runs in terms of inertia.
'''

######for now hisvec of all train and test data also obtained
    
'''
    we have the divided training datasets in to 5
    in that we train on 4 and test on 1
    so 5 iteration

    for each iteration take the 4 selected sets and find the histogram vectors of them 
    then normalize
    thrn scale
    the vector
    dump each vector of each feel in to that i th iteration's feeldump file

    have 9 C values of SVM.SVC
    for each
        we then have to run a loop 5 iterations 

        for each iteration 
        train on 4 and test on 1
        find 
        its accuracy

        and get cumulative accuracy for this 5
        that is accuracy for that Ki,C pair
    find the best C

    store it 
    with accuracy results

    then repeat for all Ki
'''
'''
    C = [0.01,0.1,1,10,100]
    
    ##### Training

    #### for each c repeat train and test and 

    for ci in C:   #for each C
        for inte in range(5):           ## for each of 5 cases
            train_X = []
            train_y = []
            for feeli in feel:
                for itee in range(5):
                    if itee != skip[inte]:
                        pkl= open(feeli+"MSh2v"+str(itee)+".pkl","rb")
                        temp = pickle.load(pkl)
                        for t in temp[1]:
                            train_y.append(temp[0])
                            train_X.append(t)                           ## loading vecs and labels  
                        pkl.close()
                        print ("done loading vecs for " + feeli +str(itee))
            svc_ = svm.SVC(kernel = "rbf", C=ci)
            ### chi2 kernel needs memory to be computed so use rbf for now
            ### Memory error with chi2kernel checkout
            print("SVM Training "+ str(svc_.kernel)+"SVMmodel"+str(inte)+"c"+str(ci)+"kMS"+str(ki))
            #print(train_X)
            svc_.fit(train_X,train_y)  

            save_obj(svc_,"SVMmodel"+str(inte)+"c"+str(ci)+"kMS"+str(ki))
    
    ### testing part to get optimal C
    #for ci in C:
        avgALL5 = []
        for inte in range(5):           ## for each of 5 cases
            SVMM = load_obj("SVMmodel"+str(inte)+"c"+str(ci)+"kMS"+str(ki))
            count = [0,0,0,0,0,0]
            scount = [0,0,0,0,0,0]
            fi = 0
            for feeli in feel:
                skk = open(feeli+"MSh2v"+str(skip[inte])+".pkl","rb")           
                temp = pickle.load(skk)
                count[fi]=len(temp[1])
                for t in temp[1]:
                    #train_y.append(temp[0])
                    index = SVMM.predict(t)
                    if index[0] == temp[0]:
                        scount[fi]=scount[fi]+1
                skk.close()
                print ("done loading vecs for " +feeli+"  "+str((scount[fi]/count[fi])*100))
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
'''
    