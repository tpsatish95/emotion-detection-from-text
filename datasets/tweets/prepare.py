# Author : Satish Palaniappan
__author__ = "Satish Palaniappan"

import pickle
import sys
sys.path.append("/mnt/RecreationHUB/MultiDomain Sentiment Classifier/script/")
from twokenize import *
import re

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def simpleProcess(text,emo):
	text = text.lower().strip()
	line = re.sub(Url_RE,"",text)
	# Testing with emo tags :P
	# for tag in stopEmoTags[emo]:
	# 	line = line.replace(tag," ")
	line = re.sub(r"#","",line)
	temp = ""
	for w in tokenize(line):
		if "@" in w:
			temp =u" ".join([temp,"@user"])
		else:
			temp =u" ".join([temp,w.strip()])
	return temp

lines = open("emotionHashtags.txt","r").readlines()

stopEmoTags = dict()
for line in lines:
	emo, tags = line.split(":")
	tags =[t.strip() for t in tags.split(",")]
	stopEmoTags[emo.strip()] = tags

print("Stop Tags Loaded !")


tags = ["joy","anger","sadness","love","fear","surprise"]

for trte in ["Tr","Te"]:
	for t in tags:
		tweetObj = load_obj(trte + "/Master"+t+trte)
		tweetObj = [simpleProcess(tweet,t) for tweet in tweetObj]
		print(tweetObj[1:100])
		print("\n\n\n")
		save_obj(tweetObj,"stopHashRemoved"+trte+"/Master"+t+trte+"P")
