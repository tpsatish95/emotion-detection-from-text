import cmath
import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier


def get_words_in_dataset(dataset):
    all_words = []
    for (words, sentiment) in dataset:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def read_datasets(fname, t_type):
    data = []
    f = open(fname, 'r')
    line = f.readline()
    while line != '':
        data.append([line, t_type])
        line = f.readline()
    f.close()
    return data


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features


def classify_dataset(data):
    return \
        classifier.classify(extract_features(nltk.word_tokenize(data)))


# read in joy , disgust, sadness, shame, anger, guilt, fear training dataset
joy_feel= read_datasets('joy.txt', 'joy')
disgust_feel = read_datasets('disgust.txt', 'disgust')
sadness_feel = read_datasets('sadness.txt', 'sadness')
anger_feel = read_datasets('anger.txt', 'anger')
guilt_feel = read_datasets('guilt.txt', 'guilt')
fear_feel = read_datasets('fear.txt', 'fear')

# filter away words that are less than 3 letters to form the training data
data = []
for (words, sentiment) in joy_feel + disgust_feel + sadness_feel + anger_feel + guilt_feel + fear_feel:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    data.append((words_filtered, sentiment))


# extract the word features out from the training data
word_features = get_word_features(\
                    get_words_in_dataset(data))



# nltk.classify.util.apply_features(feature_func, toks, labeled=None)[source]
# Use the LazyMap class to construct a lazy list-like object that is analogous to map(feature_func, toks). In particular, if labeled=False, then the returned list-like object’s values are equal to:

# [feature_func(tok) for tok in toks]
# If labeled=True, then the returned list-like object’s values are equal to:

# [(feature_func(tok), label) for (tok, label) in toks]
# The primary purpose of this function is to avoid the memory overhead involved in storing all the featuresets for every token in a corpus. Instead, these featuresets are constructed lazily, as-needed. The reduction in memory overhead can be especially significant when the underlying list of tokens is itself lazy (as is the case with many corpus readers).

# Parameters: 
# feature_func – The function that will be applied to each token. It should return a featureset – i.e., a dict mapping feature names to feature values.
# toks – The list of tokens to which feature_func should be applied. If labeled=True, then the list elements will be passed directly to feature_func(). If labeled=False, then the list elements should be tuples (tok,label), and tok will be passed to feature_func().
# labeled – If true, then toks contains labeled tokens – i.e., tuples of the form (tok, label). (Default: auto-detect based on types.)



# get the training set and train the Naive Bayes Classifier
training_set = nltk.classify.util.apply_features(extract_features, data)
# refer to saved html page outside !
classifier = NaiveBayesClassifier.train(training_set)


# read in the test tweets and check accuracy
# to add your own test tweets, add them in the respective files

total = []
for i in range(6):
    total.append(0)

test_data = read_datasets('joy_test.txt', 'joy')
total[0] =len(test_data)
test_data.extend(read_datasets('sadness_test.txt', 'sadness'))
total[1] =len(test_data) - total[0]
test_data.extend(read_datasets('disgust_test.txt', 'disgust'))
total[2] =len(test_data) - total[0] - total[1]
test_data.extend(read_datasets('anger_test.txt', 'anger'))
total[3] =len(test_data) - total[0] - total[1] - total[2]
test_data.extend(read_datasets('guilt_test.txt', 'guilt'))
total[4] =len(test_data) - total[0] - total[1] - total[2] - total[3]
test_data.extend(read_datasets('fear_test.txt', 'fear'))
total[5] =len(test_data) - total[0] - total[1] - total[2] - total[3] - total[4]


accuracy = []


for i in range(6):
	accuracy.append(0) 

emo = ["joy","disgust","sadness","anger","guilt","fear"]
num =[0,1,2,3,4,5]

emdict =dict(zip(emo,num))

prec = [0,0,0,0,0,0]

for data in test_data:
        result = classify_dataset(data[0])
        if result == "joy" and result == data[1]:
            accuracy[0]+=1
        else:
            prec[emdict[result]]+=1
        if result == "disgust" and result == data[1]:
            accuracy[1]+=1
        else:
            prec[emdict[result]]+=1 
        if result == "sadness" and result == data[1]:
            accuracy[2]+=1 
        else:
            prec[emdict[result]]+=1 
        if result == "anger" and result == data[1]:
            accuracy[3]+=1 
        else:
            prec[emdict[result]]+=1 
        if result == "guilt" and result == data[1]:
            accuracy[4]+=1 
        else:
            prec[emdict[result]]+=1 
        if result == "fear" and result == data[1]:
            accuracy[5]+=1 
        else:
            prec[emdict[result]]+=1  


tota = 0
tot = 0

for acu in accuracy:
    tota += acu
for t in total:
    tot += t
    
print('Total accuracy: %f%% (%d/20).' % (tota / tot * 100, tota))
print('Total accuracy - joy: %f%% (%d/20).' % (accuracy[0] / total[0] * 100, accuracy[0]))
print('Total accuracy - disgust: %f%% (%d/20).' % (accuracy[1] / total[1]* 100, accuracy[1]))
print('Total accuracy - sadness %f%% (%d/20).' % (accuracy[2] / total[2]* 100, accuracy[2]))
print('Total accuracy - anger %f%% (%d/20).' % (accuracy[3] / total[3]* 100, accuracy[3]))
print('Total accuracy - guilt %f%% (%d/20).' % (accuracy[4] / total[4] * 100, accuracy[4]))
print('Total accuracy - fear %f%% (%d/20).' % (accuracy[5] / total[5] * 100, accuracy[5]))



print('Total accuracy - joy: %f%%.' % (accuracy[0] /(accuracy[0]+prec[0]) * 100))
print('Total accuracy - disgust: %f%% .' % (accuracy[1] / (accuracy[1]+prec[1])* 100))
print('Total accuracy - sadness %f%% .' % (accuracy[2] / (accuracy[2]+prec[2])* 100))
print('Total accuracy - anger %f%% .' % (accuracy[3] / (accuracy[3]+prec[3])* 100))
print('Total accuracy - guilt %f%% .' % (accuracy[4] / (accuracy[4]+prec[4]) * 100))
print('Total accuracy - fear %f%% .' % ((accuracy[5] / (accuracy[5]+prec[5]) )* 100))




print('Total accuracy - joy: %f%% (%d/20).' % (cmath.sqrt(accuracy[0]/total[0] *(accuracy[0]/(accuracy[0]+prec[0]))), accuracy[0]))
print('Total accuracy - disgust: %f%% (%d/20).' % (cmath.sqrt(accuracy[1]/total[1] *(accuracy[0]/ (accuracy[1]+prec[1]))), accuracy[1]))
print('Total accuracy - sadness %f%% (%d/20).' % (cmath.sqrt(accuracy[2]/total[2] *(accuracy[0]/ (accuracy[2]+prec[2]))), accuracy[2]))
print('Total accuracy - anger %f%% (%d/20).' % (cmath.sqrt(accuracy[3]/total[3] * (accuracy[0]/(accuracy[3]+prec[3]))), accuracy[3]))
print('Total accuracy - guilt %f%% (%d/20).' % (cmath.sqrt(accuracy[4]/total[4] *(accuracy[0]/(accuracy[4]+prec[4]))), accuracy[4]))
print('Total accuracy - fear %f%% (%d/20).' % (cmath.sqrt(accuracy[5]/total[5] * (accuracy[0]/(accuracy[5]+prec[5]))), accuracy[5]))


