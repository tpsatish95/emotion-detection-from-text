from os import listdir
import pickle


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

tags = ["joy","anger","sadness","love","fear","surprise"]
for t in tags:
	files = listdir("Tweets/"+t)
	Master = []
	for f in files:
		f = f.replace(".pkl","")
		temp = load_obj("Tweets/"+t+"/"+f)
		Master.extend(temp)
	save_obj(Master,"Tr/Master"+t+"Tr")
	#Test
	print(len(Master))
	print(Master[23])