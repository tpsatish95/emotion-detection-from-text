import tweepy
from os import listdir
import pickle

# API Auth
API_KEY = "UL3Q8hMA74wRB5RykuCnCJxkm"
API_SECRET = "WJuehel0rp0wPq8T2MdULr3Hxb21DSZfZXI94oqkV6U0zz00Db"
ACCESS_TOKEN = "1970308164-GCFz3fDbgt95gs9VkbtHi0yzhTN9f8FjfbE2Ciw"
ACCESS_TOKEN_SECRET = "jZ1KL5BbHlqIEsM4qSqN2fmvu0vN4cIay2vQxy9mE6xgP"

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)
# API Auth over

print("Authenticated !!!")

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
def chunks(l, n, maxI):
    return [l[i:i + n] for i in range(n*maxI, len(l), n)]
def goto(linenum):
    global line
    line = linenum


files = listdir("Subset Train ID(100,000)(6 Emo)/")

for f in files:
	f = f.replace(".pkl","")
	listID = load_obj("Subset Train ID(100,000)(6 Emo)/"+f)

	# for resume downlaod
	fi = listdir("Subset Train ID(100,000)(6 Emo)/Tweets/"+f+"/")
	done = []
	for i in fi:
		done.append(int(i.replace(".pkl","")))
	if not done:
		maxID=0
	else:
		maxID = max(done)
	i = maxID+1
	#######
	for sub in chunks(listID,100,maxID):
		try:
			statuses = api.statuses_lookup(sub)
			statuses = [s.text for s in statuses]
			save_obj(statuses,"Subset Train ID(100,000)(6 Emo)/Tweets/"+f+"/"+str(i))
			print("Chunk " +f+str(i)+ " Done.")
			i+=1
		except:
			print("Retrying...")
			goto(51)


# Testing....
# statuses = api.statuses_lookup(["147750646089658368","135724218678657024","137348837869240320"])

# for s in statuses:
# 	print(s.text)