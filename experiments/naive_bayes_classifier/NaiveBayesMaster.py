import cmath
import nltk
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_words_in_dataset(dataset):
    all_words = []
    for (words, sentiment) in dataset:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = []
    for wk in wordlist.keys():
        if wordlist[wk] > 5:    
            word_features.append(wk) 
    return word_features


def read_datasets(flist, t_type):
    data = []
    for l in flist:
        data.append([l, t_type])
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
joy_feel= read_datasets(load_obj("../DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterjoyPTr"), 'joy')
anger_feel = read_datasets(load_obj("../DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterangerPTr"), 'anger')
sadness_feel = read_datasets(load_obj("../DATA/Preproccess Tools/Preprocessor/ProcessedTr/MastersadnessPTr"), 'sadness')
surprise_feel = read_datasets(load_obj("../DATA/Preproccess Tools/Preprocessor/ProcessedTr/MastersurprisePTr"), 'surprise')
love_feel = read_datasets(load_obj("../DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterlovePTr"), 'love')
fear_feel = read_datasets(load_obj("../DATA/Preproccess Tools/Preprocessor/ProcessedTr/MasterfearPTr"), 'fear')

# filter away words that are less than 3 letters to form the training data
data = []
for (sentence, sentiment) in joy_feel + love_feel + sadness_feel + anger_feel + surprise_feel + fear_feel:
    words_filtered = [e for e in sentence.split()]
    data.append((words_filtered, sentiment))


# extract the word features out from the training data
word_features = get_word_features(get_words_in_dataset(data))
print("Loading and Formatting Done.")


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
print(training_set)
print("Unigram FreqDist Features Extarcted.")
#### TF-IDF NOT applied


### using nltk for NB is very slow as it is only for learning purposes USE SCIKIT-LEARN
# cv = cross_validation.KFold(len(training_set), n_folds=10)
# print("Train Data Split Up.")

# print("Training and Validating....")

# i=1
# for traincv, testcv in cv:
#     print("Iteration No. "+str(i)+" of 10")
#     classifier = NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
#     Accuracy = nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
#     Precision=nltk.classify.util.precision(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
#     Recall=nltk.classify.util.recall(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
#     FMeasure=nltk.classify.util.f_measure(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
#     print ('Accuracy:'+ str(Accuracy))
#     print ('Precision:'+ str(Precision))
#     print ('Recall:'+ str(Recall))
#     print ('FMeasure:'+ str(FMeasure))
#     save_obj(classifier,"MNB"+str(i)+"F"+str(FMeasure))
#     i+=1
c = input("Continue!?")
c = input("Continue!?")