### pickling !!!!!
# see pickle
import pickle
import numpy as np
from sklearn import svm

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

Dump = ["joyD.pkl","angerD.pkl","loveD.pkl","sadnessD.pkl","fearD.pkl","surpriseD.pkl"]
labels  = ["joy","anger","love","sadness","fear","surprise"]
lable_nums = [0,1,2,3,4,5]

emo_Dict = dict(zip(lable_nums,labels))

train_X = []
train_y = []
i=0

for d in Dump:
	DumpP = open(d,"rb")
	curEMO = labels[i].split("O")[0]
	print(curEMO)
	temp = pickle.load(DumpP)
	for t in temp[1]:
		train_y.append(temp[0])
		train_X.append(t)                           ## loading vecs and labels  
	DumpP.close()
	print ("done loading vecs for " + curEMO)
	i=i+1

svc_ = svm.SVC(kernel = "rbf")
print("SVM Training "+ svc_.kernel)
#print(train_X)
svc_.fit(train_X,train_y)  

save_obj(svc_,"SVMmodel")

print("Model Saved")