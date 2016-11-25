### pickling !!!!!
# see pickle
import pickle
import numpy as np
import nltk
import string
from nltk.stem.lancaster import LancasterStemmer
from sklearn import svm
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

#print (LookUpDict.values())

hist, bin_edges = np.histogram(list(LookUpDict.values()), bins=range(13))
#plt.hist(list(LookUpDict.values()), bins=range(11))
#plt.show()
print(hist)

labels  = ["joy","anger","love","sadness","fear","surprise"]
files = ["HAPPY.txt","ANGRY.txt","LOVE.txt","SADNESS.txt","FEAR.txt","SURPRISE.txt"]


lable_nums = [0,1,2,3,4,5]

emo_Dict = dict(zip(lable_nums,labels))

i=0

count  = [0,0,0,0,0,0]
scount  = [0,0,0,0,0,0]

exclude = set(string.punctuation)

svmModel = load_obj("SVMmodel")

for fi in files:
	text = open("Test/"+fi,"rU")
	lines  = text.readlines()
	hisvec=[]
	for line in lines:
		tokens = []
		words = nltk.word_tokenize(line)
		Uwords = [w for w in words if w not in exclude]

		for word in Uwords:
			tokens.append(word.strip().lower())  #stemmer.stem() removed
			#print(tokens)
		#tokens = list(set(tokens)- set(nltk.corpus.stopwords.words('english')))
		#print(tokens)
		Svec = []
		for token in tokens:
			try:
				Svec.append(LookUpDict[token])
			except:
				###Svec.append(-1)
				pass  ##token not found so undefined bin we dont have training data for undefined so neglect
		histoVec, bin_edges = np.histogram(Svec, bins=range(13))	
		#print(histoVec)	#13 dimension vec of sentence
		histoVec = [float(his) for his in histoVec]
		hisvec.append(histoVec)
		## vector obtained
		# index = svmModel.predict(histoVec)
		# count[i]=count[i]+1
		# #print(emo_Dict[index[0]])
		# if emo_Dict[index[0]] == labels[i]:
		# 	scount[i]=scount[i]+1
	h2vN = preprocessing.normalize(hisvec)
	h2vS = min_max_scaler.fit_transform(h2vN)
	a=np.empty(len(h2vS))
	a.fill(i)
	print("Accuracy for "+labels[i]+"===   "+str(svmModel.score(h2vS,a)))
	i=i+1