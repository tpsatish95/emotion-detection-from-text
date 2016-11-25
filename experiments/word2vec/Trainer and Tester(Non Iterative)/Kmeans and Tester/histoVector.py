### pickling !!!!!
# see pickle
import pickle
import numpy as np
import nltk
import string
from nltk.stem.lancaster import LancasterStemmer
from sklearn import cluster, datasets, preprocessing

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

stemmer = LancasterStemmer()
min_max_scaler = preprocessing.MinMaxScaler()


LookUpDict = load_obj("Cluster_lookUPTable")

# #print (LookUpDict.values())

hist, bin_edges = np.histogram(list(LookUpDict.values()), bins=range(13))
# #plt.hist(list(LookUpDict.values()), bins=range(11))
# #plt.show()
print(hist)

files = ["joyO.txt","angerO.txt","loveO.txt","sadnessO.txt","fearO.txt","surpriseO.txt"]
Dump = ["joyD.pkl","angerD.pkl","loveD.pkl","sadnessD.pkl","fearD.pkl","surpriseD.pkl"]
labels  = ["joy","anger","love","sadness","fear","surprise"]

lable_nums = [0,1,2,3,4,5]

emo_Dict = dict(zip(lable_nums,labels))

i=0

exclude = set(string.punctuation)

#fe = open("s.txt","w")

for f,d in zip(files,Dump):
	cF = open(f,"rU")
	DumpP = open(d,"wb")
	curEMO = labels[i].split("O")[0]
	print(curEMO)
	lines = cF.readlines()
	hisvec=[]
	for line in lines:
		tokens = []
		words = nltk.word_tokenize(line)
		Uwords = [w for w in words if w not in exclude]
		for word in Uwords:
			tokens.append(word.strip().lower())  #stemmer.stem() removed
		#print(tokens)
		Svec = []
		for token in tokens:
			try:
				Svec.append(LookUpDict[token])
			except:
				###Svec.append(-1)
				pass  ##token not found so undefined bin we dont have training data for undefined so neglect

		histoVec, bin_edges = np.histogram(Svec, bins=range(13))	
		histoVec = [float(his) for his in histoVec]
		hisvec.append(histoVec)
		#print(histoVec)	#30 dimension vec of sentence
		#svmModel = load_obj("SVMmodel")
		#index = svmModel.predict(histoVec)
		#fe.write(str(index)
	h2vN = preprocessing.normalize(hisvec)
	h2vS = min_max_scaler.fit_transform(h2vN)

	pickle.dump([lable_nums[i], h2vS], DumpP, pickle.HIGHEST_PROTOCOL)
	DumpP.close()
	cF.close()
	i=i+1
#fe.close()
