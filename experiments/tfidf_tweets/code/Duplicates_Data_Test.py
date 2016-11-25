import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

# read in joy , disgust, sadness, shame, anger, guilt, fear training dataset
joy_feeltr= load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterjoyPTr")
anger_feeltr = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterangerPTr")
sadness_feeltr = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MastersadnessPTr")
surprise_feeltr = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MastersurprisePTr")
love_feeltr = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterlovePTr")
fear_feeltr = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterfearPTr")



# read in joy , disgust, sadness, shame, anger, guilt, fear test dataset
joy_feelte= load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterjoyPTe")
anger_feelte = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterangerPTe")
sadness_feelte = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MastersadnessPTe")
surprise_feelte = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MastersurprisePTe")
love_feelte = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterlovePTe")
fear_feelte = load_obj("DATA/Preproccess Tools/Preprocessor/ProcessedTe/MasterfearPTe")

DesignMatrix = [joy_feeltr,anger_feeltr,sadness_feeltr,surprise_feeltr,love_feeltr,fear_feeltr]
TestMatrix = [joy_feelte,anger_feelte,sadness_feelte,surprise_feelte,love_feelte,fear_feelte]

tags = ["joy","anger","sadness","love","fear","surprise"]

i=0
tot=0
ttot=0
print("Train set Duplicates:")
for j in DesignMatrix:
	print(tags[i]+": "+str((len(j)-len(list(set(j))))))
	# print("With Duplicates: "+str(len(j)))
	# print("Without Duplicates: "+str(len(list(set(j)))))
	tot+= (len(j)-len(list(set(j))))
	ttot+=len(j)
	i+=1
print("Total Duplicates: " + str(tot)  +" of " +str(ttot))

i=0
tot=0
ttot=0
print("Test set Duplicates:")
for j in TestMatrix:
	print(tags[i]+": "+str((len(j)-len(list(set(j))))))
	# print("With Duplicates: "+str(len(j)))
	# print("Without Duplicates: "+str(len(list(set(j)))))
	tot+= (len(j)-len(list(set(j))))
	ttot+=len(j)
	i+=1
print("Total Duplicates : " +str(tot) +" of " +str(ttot))

test = []
train = []
for t in DesignMatrix:
	train.extend(t)
for t in TestMatrix:
	test.extend(t)

train = set(train)
test = set(test)

print("Total Unique Train: " +str(len(train)))
print("Total Unique Test: " +str(len(test)))
print("Tweets common to Test and Train: " +str(len(train.intersection(test))))

print("Total Tweets in dataset:")
print(len(train.union(test)))
print("Total Unique Tweets in dataset:")
print(len(train.union(test))-len(train.intersection(test)))