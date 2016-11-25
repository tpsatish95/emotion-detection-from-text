import pickle


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

IdtoEmo = load_obj("IdtoEmoTr")

joy = list()
sadness = list()
anger = list()
fear = list()
love = list()
surprise = list()
thankfulness = list()


for i in IdtoEmo.keys():
	if IdtoEmo[i] == "joy":
		joy.append(i)
	elif IdtoEmo[i] == "anger":
		anger.append(i)
	elif IdtoEmo[i] == "sadness":
		sadness.append(i)
	elif IdtoEmo[i] == "love":
		love.append(i)
	elif IdtoEmo[i] == "thankfulness":
		thankfulness.append(i)
	elif IdtoEmo[i] == "fear":
		fear.append(i)
	elif IdtoEmo[i] == "surprise":
		surprise.append(i)

lists = [joy,anger,sadness,love,fear,surprise]
tags = ["joy","anger","sadness","love","fear","surprise"]

j=0 # for tags
for li in lists:
	#Randomize
	li = set(li)
	li = list(li)
	i=0 # for 100000 count
	temp = []
	for l in li:
		temp.append(l)
		i+=1
		if i>100000:
			break
	save_obj(temp,tags[j])
	j+=1