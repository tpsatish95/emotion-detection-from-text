import nltk
import re
import string


exclude = set(string.punctuation)
feel = ["ANGER","JOY","FEAR","LOVE","SURPRISE","SADNESS"]


docs = []
for fe in feel:
	f =  open(fe+"_Phrases.txt","r")
	
	lines = f.readlines()
	f.close()
	for line in lines:
		feell=[]
		line = line.lower()
		words = re.findall(r"[\w']+|[.,/;=?!$%:#\"]",line)
		Uwords = [w.strip() for w in words if w not in exclude]
		for word in Uwords:
			feell.append(word)
		docs.append(feell)


### docs is the 2d list u asked for