import string
import re
import nltk
import numpy
import gensim
from gensim import corpora, similarities, models

punctuation_string=['.',',']
stoplist=['I','in','the','my']
def getTopicForQuery (question):
    temp = question.lower()
    print("a.",question)
    for i in range(len(punctuation_string)):
        temp = temp.replace(punctuation_string[i], '')
        print("b.",temp)

    words = re.findall(r'\w+', temp, flags = re.UNICODE | re.LOCALE)
    print("c.",words)

    important_words = []
    important_words = filter(lambda x: x not in stoplist, words)
    print("d.",important_words)

    dictionary = corpora.Dictionary.load('questions.dict')

    ques_vec = []
    ques_vec = dictionary.doc2bow(important_words)
    print("e.",ques_vec)

    topic_vec = []
    topic_vec = lda[ques_vec]
    print("f.",topic_vec)

    word_count_array = numpy.empty((len(topic_vec), 2), dtype = numpy.object)
    print("g.",word_count_array)
    for i in range(len(topic_vec)):
        word_count_array[i, 0] = topic_vec[i][0]
        word_count_array[i, 1] = topic_vec[i][1]
        print("h.",word_count_array)

    idx = numpy.argsort(word_count_array[:, 1])
    print("i.",idx)
    idx = idx[::-1]
    print("j.",idx)
    word_count_array = word_count_array[idx]
    print("k.",word_count_array)

    final = []
    final = lda.print_topic(word_count_array[0, 0], 1)
    print("l.",final)

    
    question_topic = final.split('*') ## as format is like "probability * topic"
    print("m.",question_topic)

    return question_topic[1]

##Text Preprocessing is done here using nltk
##Saving of the dictionary and corpus is done here
##final_text contains the tokens of all the documents


final_text = [['I','love','the','atmosphere','of','this','place'],['I','love','my','my','mother','and','father'],['I', 'love', 'my', 'dog', 'very', 'much'],['I' ,'enjoy' ,'playing' ,'in' ,'the' ,'rain']]
dictionary = corpora.Dictionary(final_text)
print ("1.",dictionary)
dictionary.save('questions.dict');
corpus = [dictionary.doc2bow(text) for text in final_text]
print ("2.",corpus)
print("3.",corpora.MmCorpus.serialize('questions.mm', corpus))
print("4.",corpora.SvmLightCorpus.serialize('questions.svmlight', corpus))
print("5.",corpora.BleiCorpus.serialize('questions.lda-c', corpus))
print("6.",corpora.LowCorpus.serialize('questions.low', corpus))

##Then the dictionary and corpus can be used to train using LDA

mm = corpora.MmCorpus('questions.mm')
print ("7.",mm)
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=100, update_every=0, chunksize=19188, passes=20)
print("8.",lda)


print("*************")
print(getTopicForQuery("I like atmosphere and dog very much."))
