### pickling !!!!!
# see pickle

#pickle has protocol error
### py 3 has a higher protocol py 2 has
###pickle.dump(your_object, your_file, protocol=2)

from sklearn import cluster, datasets, preprocessing
import pickle
import numpy as np
#import nltk
import string
import re
#from nltk.stem.lancaster import LancasterStemmer
import gensim
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


KM = load_obj("bestKMEANSforK12")
model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True)

ki=12

#print (LookUpDict.values())

#hist, bin_edges = np.histogram(list(LookUpDict.values()), bins=range(13))
#plt.hist(list(LookUpDict.values()), bins=range(11))
#plt.show()
#print(hist)

#files = ["joyO.txt","angerO.txt","disgustO.txt","sadnessO.txt","fearO.txt","surpriseO.txt"]
#Dump = ["joyD.pkl","angerD.pkl","disgustD.pkl","sadnessD.pkl","fearD.pkl","surpriseD.pkl"]
#labels  = ["joy","anger","disgust","sadness","fear","surprise"]

feel = ["ANGER","JOY","FEAR","LOVE","SURPRISE","SADNESS"]
lable_nums = [0,1,2,3,4,5]

#emo_Dict = dict(zip(lable_nums,labels))


exclude = set(string.punctuation)

#fe = open("s.txt","w")

skip = [4,3,2,1,0]

i=0

min_max_scaler = preprocessing.MinMaxScaler()


for feeli in feel:	
	for ite in range(5):
		hisvec = []
		hisvecS = []
		DumpP = open(feeli+"h2v"+str(ite)+".pkl","wb")
		DumpPS = open(feeli+"h2vSkip"+str(ite)+".pkl","wb")
		for sp in range(5):
			w = open("SPlit/"+feeli+str(sp)+".txt","r")
			print(feeli+str(sp))
			lines = w.readlines()
			w.close()
			for line in lines:
				tokens = []
				words = re.findall(r"[\w']+|[.,/;=?!$%]",line.lower())
				Uwords = [w for w in words if w not in exclude]
				for word in Uwords:
					tokens.append(word.strip().lower())  #stemmer.stem() removed
				#print(tokens)
				Svec = []
				for token in tokens:
					try:
						Svec.append(KM.predict(model[token]).tolist()[0])
					except:
						###Svec.append(-1)
						pass  ##token not found so undefined bin we dont have training data for undefined so neglect
				histoVec, bin_edges = np.histogram(Svec, bins=range(ki+1))
				histoVec = [float(his) for his in histoVec]
				if sp != skip[ite]:
					hisvec.append(histoVec)
				else:
					hisvecS.append(histoVec)
					#print (histoVec)
		h2vN = preprocessing.normalize(hisvec)
		#h2vS = preprocessing.scale(h2vN)
		h2vS = min_max_scaler.fit_transform(h2vN)
		h2vNS = preprocessing.normalize(hisvecS)
		#h2vSS = preprocessing.scale(h2vNS)			
		h2vSS = min_max_scaler.fit_transform(h2vNS)

		##### the chi2_kernel accepts only positive values
		##### scales feature vector between 0 and 1



		#print(histoVec)	#30 dimension vec of sentence
		#svmModel = load_obj("SVMmodel")
		#index = svmModel.predict(histoVec)
		#fe.write(str(index))
		
		"""
		h2vS has the entire 2d array of vectors of a file (4 files) each row is a sentence  
		you will get 5 pickle files for each emotion to iterate for all C values of SVM time
		"""
		pickle.dump([lable_nums[i], h2vS],DumpP,  protocol=2)
		pickle.dump([lable_nums[i], h2vSS],DumpPS,  protocol=2)
		print(feeli+ str(ite) +" and Skip for Test also Dumped!!")
		DumpP.close()
		DumpPS.close()
	i=i+1

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
C = [0.001,0.01,0.1,1,10,100,1000]
'''
##### Training

#### for each c repeat train and test and 

for ci in C:   #for each C
	for inte in range(5):			## for each of 5 cases
		train_X = []
		train_y = []
		for feeli in feel:
			pkl= open(feeli+"h2v"+str(inte)+".pkl","rb")
			while 1:
				try:
					temp = pickle.load(pkl)
					for t in temp[1]:
						train_y.append(temp[0])
						train_X.append(t)							## loading vecs and labels	
				except:
					pkl.close()
					print ("done loading vecs for " + feeli)
					break
		svc_ = svm.SVC(kernel = "rbf", C=ci)
		### chi2 kernel needs memory to be computed so use rbf for now
		### Memory error with chi2kernel checkout
		print("SVM Training "+ str(svc_.kernel)+"SVMmodel"+str(inte)+"c"+str(ci)+"k"+str(ki))
		#print(train_X)
		svc_.fit(train_X,train_y)  

		save_obj(svc_,"SVMmodel"+str(inte)+"c"+str(ci)+"k"+str(ki))
'''

### testing part to get optimal C
for ci in C:
	avgALL5 = []
	for inte in range(5):			## for each of 5 cases
		SVMM = load_obj("SVMmodel"+str(inte)+"c"+str(ci)+"k"+str(ki))
		count = [0,0,0,0,0,0]
		scount = [0,0,0,0,0,0]
		fi = 0
		for feeli in feel:
			skk = open(feeli+"h2vSkip"+str(inte)+".pkl","rb")			
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
		avgg.add(t/5)
	store = open("R"+str(ki)+"C"+str(ci)+".txt","w")
	for eemo in range(6):
		store.write("EMO "+str(feel[eemo])+" Accu "+str(avgg[eemo])+"\n")
	store.close()
