import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn import cluster, datasets, preprocessing
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import pickle
import gensim
import time
import re
import twokenize

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = clf.predict(X_test)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        # print("top 10 keywords per class:")
        # for i, category in enumerate(categories):
        #     top10 = np.argsort(clf.coef_[i])[-10:]
        #     print("%s: %s" % (category, " ".join(feature_names[top10]).encode("utf-8")))
        # print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred,target_names=categories))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def read_datasets(flist, t_type):
    data = []
    for l in flist:
        data.append([l, t_type])
    return data

min_max_scaler = preprocessing.MinMaxScaler()

model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)
model.init_sims(replace=True)

Vocab = []
W2vX=[]
for word in model.vocab:
    W2vX.append(model[word])
    Vocab.append(word)
W2vX = np.array(W2vX)

##### minimizing intertia or energy for a single cluster size k

# ##intialize k
# k = [7000,14000]
# '''100(k = 700), 50(k = 1400), 10, 5 words per cluster'''
feel = ["joy","anger","sadness","surprise","love","fear"]
lable_nums = [0,1,2,3,4,5]

Cluster_Model_Path = "ClusterModel/"

# f = 1

# for ki in k:
ki = 14259
# # KM = ""
# # if f == 1:
# #     KM = load_obj(Cluster_Model_Path + str(ki) + "KMeans")
# #     f = 0
# # else:
# print("K is " +str(ki))
# t0 = time.time()
# # kmeans = cluster.KMeans(n_clusters=ki,n_init=100).fit(X)
# kmeans = cluster.MiniBatchKMeans(n_clusters=ki,n_init=20,batch_size = ki*3).fit(W2vX)
# print(str(time.time()-t0))
# print(kmeans.inertia_)
# print("Done for K " +str(ki))
# ##### save best for future use
# save_obj(kmeans,Cluster_Model_Path + str(ki) + "KMeans")
# KM = kmeans

# Cluster_lookUP = dict()
# print("Building Label Dictionary !")
# #Slow
# # for word in model.vocab:
# # 	Cluster_lookUP[word] = KM.predict(model[word])[0]

# #Fast
# Cluster_lookUP = dict(zip(Vocab, KM.labels_))


# using classes.C (Own Code) to get 14259 Clusters faster
print("Building Label Dictionary !")
Cluster_lookUP = dict()
for l in open("W2Vclasses.sorted.txt","r").readlines():
    word, label = l.split()
    Cluster_lookUP[word] = label

print("Building Histograms !")
for feeli in feel:
    hisvec = []
    lines = list(set(load_obj("DATA/Preprocess Tools/Preprocessor/stopHashRemovedTr/Master"+feeli+"TrP")))[:1000]
    ti = time.time()
    for line in lines:
        words = line.lower().split()
        Svec = []
        for word in words:
            try:
                Svec.append(Cluster_lookUP[word.strip()])
            except:
                continue
        histoVec, bin_edges = np.histogram(Svec, bins=range(ki+1))
        histoVec = [float(his) for his in histoVec]
        hisvec.append(histoVec)
    print(str(time.time()-ti))
    h2vN = preprocessing.normalize(hisvec)
    h2vS = min_max_scaler.fit_transform(h2vN)


    save_obj(h2vS,Cluster_Model_Path + feeli + "Histogram")
    print(feeli+" Dumped!!")

train_X = []
train_y = []
joy_feel= read_datasets(load_obj(Cluster_Model_Path + "joy" + "Histogram"), 'joy')
anger_feel = read_datasets(load_obj(Cluster_Model_Path + "anger" + "Histogram"), 'anger')
sadness_feel = read_datasets(load_obj(Cluster_Model_Path + "sadness" + "Histogram"), 'sadness')
surprise_feel = read_datasets(load_obj(Cluster_Model_Path + "surprise" + "Histogram"), 'surprise')
love_feel = read_datasets(load_obj(Cluster_Model_Path + "love" + "Histogram"), 'love')
fear_feel = read_datasets(load_obj(Cluster_Model_Path + "fear" + "Histogram"), 'fear')

DesignMatrix = joy_feel+anger_feel+sadness_feel+surprise_feel+love_feel+fear_feel

X=[]
y=[]
for i in DesignMatrix:
    X.append(i[0])
    y.append(i[1])
X = np.array(X)
y = np.array(y)
print ("Done loading Histograms!")

# C_range = 10.0 ** np.arange(-1, 1)
# # gamma_range = [0.0]
# param_grid = dict(C=C_range)
# cv = StratifiedKFold(y=train_y, n_folds=5)
# grid = GridSearchCV(LinearSVC(loss='l2', penalty="l2"), param_grid=param_grid, cv=cv)
# print("Searching the SVM Grid !")
# grid.fit(train_X, train_y)

# store = open(Cluster_Model_Path + "Result"+str(ki)+"ClusterSize.txt","w")
# store.write(str(grid.best_estimator_))
# store.write("\n"+str(grid.best_score_))
# store.close()
# save_obj(grid,Cluster_Model_Path + str(ki)+"SVCGrid")

t0 = time.time()

skf = StratifiedKFold(y, n_folds=10,shuffle=True)
print(skf)

for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('data loaded')

    categories = ["anger","fear","joy","love","sadness","surprise"]
    benchmark(LinearSVC(loss='l2', penalty="l2",dual=False, tol=1e-3))
    break

print(str(time.time()-t0))

print("Done " + str(ki))
