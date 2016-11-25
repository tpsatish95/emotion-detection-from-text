### pickling !!!!!
# see pickle

import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)





from sklearn import cluster, datasets
f = open("WFvec.txt")

vecs = f.readlines()

i=0

wL = []

Wvecs = []

t = []

tempDUMP=open("Dump.pkl","wb")

print ("Loading File!!!!")

for vec in vecs:
	t = []
	if i % 2 ==0:
		wL=wL+[vec.strip()]
	elif i % 2 ==1:
		pickle.dump([float(a.strip()) for a in vec.split()], tempDUMP, pickle.HIGHEST_PROTOCOL)
		#Wvecs.append([float(a.strip()) for a in vec.split()])
	i=i+1

print ("File read !!!!")
tempDUMP.close()

DUMP = open("Dump.pkl","rb")

#now load pkl file in to 2d arr
while 1:
	try:
		temp = pickle.load(DUMP)
		#print(tempDUMP)
		Wvecs.append(temp)
	except:
		DUMP.close()
		print ("done loading vecs")
		break
# done loading vecs



k_means = cluster.KMeans(n_clusters=12,n_init=300)
k_means.fit(Wvecs)

print("Clustered")

Cluster_lookUP = dict(zip(wL, k_means.labels_))

save_obj(Cluster_lookUP,"Cluster_lookUPTable")

print ("Saved")