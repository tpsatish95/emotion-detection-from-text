#!/usr/bin/env python
import re, random, math, collections, itertools



#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):


    #reading pre-labeled movie reviews and splitting into lines

    posSentences=[]    #initialise list
    negSentences=[]    #initialise list

    txt = open('Data/Movies/rt-polarity.pos', 'r')
    posSentences = re.split(r'\n', txt.read())

    txt = open('Data/Movies/rt-polarity.neg', 'r')
    negSentences = re.split(r'\n', txt.read())

    #reading pre-labeled Nokia reviews and splitting into lines
    posSentencesNokia=[]    #initialise list
    negSentencesNokia=[]    #initialise list

    txt = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', txt.read())

    txt = open('nokia-neg.txt', 'r')
    negSentencesNokia = re.split(r'\n', txt.read())
 
    #reading Sentiment Dictionaries
    posWordList=[]    #initialise list
    negWordList=[]    #initialise list
    txt = open('positive-words.txt', 'r')
    posWordList = re.findall(r"[a-z\-]+", txt.read())

    txt = open('negative-words.txt', 'r')
    negWordList = re.findall(r"[a-z\-]+", txt.read())

    #Create single sentiment dictionary, where words have value 1 if positive and -1 if negative:

    sentimentDictionary={} #initialise dictionary

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

    #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    sentenceTrain={}
    sentimentTest={}    

    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset, with sentiment attached to sentences:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"
    
#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0
    
    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.iteritems():
        wordList = re.findall(r"[\w']+", sentence) # get word list
        for word in wordList:
            allWordsTot += 1 # keeps count of total words in dataset
            if not dictionary.has_key(word):
                dictionary[word] = 1
            if sentiment=="positive":
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not freqPositive.has_key(word):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not freqNegative.has_key(word):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not freqNegative.has_key(word):
            freqNegative[word] = 1
        if not freqPositive.has_key(word):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot) 

        # Calculate p(word|negative)
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord, pPos):
    pNeg=1-pPos
    
    #These variables will store results (you do not need them)
    total=0
    correct=0
    totalpos=0
    totalneg=0
    correctpos=0
    correctneg=0
    
    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.iteritems():
        Words = re.findall(r"[\w']+", sentence)#collect all words


#-------------Practical 1: Write Code Here -----------------------------# 
        # At this point, Words contains the list of words in the sentence.
        # You need to implement Naive Bayes to decide the sentiment of the 
        # sentence based on the list of words.
        # Write the loop to calculate:
        #   Score(Positive) = p(Positive) *p(w1|Positive)*...*p(wn|Positive)
        #   Score(Negative) = p(Negative) *p(w1|Negative)*...*p(wn|Negative)

        # Now, using the above calculations, calculate a variable "prob" that holds the likelihood of the sentence being Positive:
        # p(Positive|w1,...wn) = Score(Positive)/(Score(Positive)+Score(Negative))
        #In short, you need to:
        # a) Write the loop to calculate the Scores
        # b) create a variable called "prob" that stores p(Positive|w1,...,wn)



#------------------------Finish Practical 1 Code--------------------------#

        #If you have written the code above, the code below will keep track of classification accuracy. Do not change the indentation, the code below is part of the for loop.
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
            else:
                correct+=0
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
            else:
                correct+=0
 
    acc=correct/float(total)
    print dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")"
    accpos=correctpos/float(totalpos)
    accneg=correctneg/float(totalneg)
    print dataName + " Accuracy (Pos)=%0.2f" % accpos + " (%d" % correctpos + "/%d" % totalpos + ")"
    print dataName + " Accuracy (Neg)=%0.2f" % accneg + " (%d" % correctneg + "/%d" % totalneg + ")\n"



# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.iteritems():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if sentimentDictionary.has_key(word):
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
            else:
                correct+=0
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
            else:
                correct+=0
 
    acc=correct/float(total)
    print dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")"
    accpos=correctpos/float(totalpos)
    accneg=correctneg/float(totalneg)
    print dataName + " Accuracy (Pos)=%0.2f" % accpos + " (%d" % correctpos + "/%d" % totalpos + ")"
    print dataName + " Accuracy (Neg)=%0.2f" % accneg + " (%d" % correctneg + "/%d" % totalneg + ")\n"




#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

#run naive bayes classifier on datasets
print "Naive Bayes"
testBayes(sentencesTrain,  "Films (Train Data)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia   (All Data)\t", pWordPos, pWordNeg, pWord,0.7)

#run sentiment dictionary based classifier on datasets
print "Sentiment Dictionary"
#testDictionary(sentencesTrain,  "Films (Train Data)\t", sentimentDictionary, -4)
#testDictionary(sentencesTest,  "Films  (Test Data)\t",  sentimentDictionary, -4)
#testDictionary(sentencesNokia, "Nokia   (All Data)\t",  sentimentDictionary, -3)

