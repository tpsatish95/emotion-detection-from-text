### pickling !!!!!
# see pickle

import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)





from sklearn import cluster, datasets, preprocessing
import numpy as np
import gensim

f = open("finv.txt","r",encoding="iso-8859-15")

vecs = f.readlines()

i=0

wL = []

Wvecs = []

t = []

model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)
model.init_sims(replace=True)

skip=0

#now load pkl file in to 2d arr
for vec in vecs:
	try:
		#print(tempDUMP)
		Wvecs.append(model[vec.strip().lower()])
		wL.append(vec.strip().lower())
	except:
		print ("no match")
		skip=skip+1

X = np.array(Wvecs)
X_Scaled_Feature_Vecs = preprocessing.scale(X)

Word2NVDict = dict(zip(wL, X_Scaled_Feature_Vecs.tolist()))

save_obj(Word2NVDict,"Word2NVDict")
# done loading vecs


#k_means = cluster.KMeans(n_clusters=12)
#k_means.fit(Wvecs)

#print("Clustered")

#Cluster_lookUP = dict(zip(wL, k_means.labels_))

#save_obj(Cluster_lookUP,"Cluster_lookUPTable")

print ("Saved")
print(str(skip))
