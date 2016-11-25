import pickle
from os import listdir

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

IDtoEmo = dict()

files = listdir("Train/")

for f in files:
	handle = open("Train/"+f,"r")
	lines = handle.readlines()
	handle.close()

	for line in lines:
		a,b = line.split()
		IDtoEmo[a.strip()] = b.strip()

save_obj(IDtoEmo,"IDtoEmoTr")

#Check
# for k in IDtoEmo.keys():
# 	print(IDtoEmo[k])
# 	break
