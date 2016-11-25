### pickling !!!!!
# see pickle


#no Plans Change
''' 
we are taking all words in the word2vec google dataset and cluster it
'''

import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)





from sklearn import cluster, datasets, preprocessing
import gensim
import numpy as np
i=0

wL = []

Wvecs = []

t = []

model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)
model.init_sims(replace=True)
min_max_scaler = preprocessing.MinMaxScaler()
X = np.array(model.syn0)
### scaling feature vecs

X_Scaled_Feature_Vecs = min_max_scaler.fit_transform(X)

# emo6 = [[vecv for vecv in line.split()] for line in open("6emo.txt").readlines()]
# print(len(emo6))
#now load pkl file in to 2d arr

print(len(X_Scaled_Feature_Vecs[0]))

k_means = cluster.KMeans(n_clusters=12,n_init=100)
k_means.fit(X_Scaled_Feature_Vecs)
print(k_means.inertia_)

print("Clustered")

Cluster_lookUP = dict(zip(model.vocab, k_means.labels_))

save_obj(Cluster_lookUP,"Cluster_lookUPTable")

print ("Saved")