from sklearn import cluster, datasets, preprocessing
import numpy as np
import gensim
import pickle



def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)



model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)


#### getting all vecs from w2v using the inbuilt syn0 list see code

X = np.array(model.syn0)

### scaling feature vecs

X_Scaled_Feature_Vecs = preprocessing.scale(X)

##### minimizing intertia or energy for a single cluster size k

##intialize k
k = [256,512,1024,2048,4096,8192]

### max K that we can reach

###MAX_K = 8192

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

##### better code


for ki in k:
	kmeans = cluster.KMeans(n_clusters=ki,n_init=100).fit(X_Scaled_Feature_Vecs)
	print(kmeans.inertia_)
	print("k===" +str(ki))
	##### save best for future use
	save_obj(kmeans,"bestKMEANSforK"+str(ki))

