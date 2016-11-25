import nltk
import nltk.classify

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about tonight\'s concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to tonight\'s concert', 'negative'),
              ('He is my enemy', 'negative')]

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
  words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
  tweets.append((words_filtered, sentiment))

print tweets

tt = [('I feel happy this morning', 'positive'),
               ('Larry is my friend', 'positive'),
               ('I do not like that man', 'negative'),
               ('My house is not great', 'negative'),
               ('Your song is annoying', 'negative')]

test_tweets = []
for (words, sentiment) in tt:
  words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
  test_tweets.append((words_filtered, sentiment))

print test_tweets


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words
 
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    print wordlist
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(
                    get_words_in_tweets(tweets))

print word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features

print extract_features(tweets[0][0])

training_set = nltk.classify.apply_features(extract_features, tweets)
test_set = nltk.classify.apply_features(extract_features, test_tweets)
    
classifier = nltk.NaiveBayesClassifier.train(training_set)
print classifier.show_most_informative_features(256)

tweet = 'I feel happy this morning'
print extract_features(tweet.split())
sentiment = classifier.classify(extract_features(tweet.split()))
print sentiment

tweet = 'Larry is my friend'
print extract_features(tweet.split())
sentiment = classifier.classify(extract_features(tweet.split()))
print sentiment

tweet = 'I do not like that man'
print extract_features(tweet.split())
sentiment = classifier.classify(extract_features(tweet.split()))
print sentiment

tweet = 'My house is not great'
print extract_features(tweet.split())
sentiment = classifier.classify(extract_features(tweet.split()))
print sentiment

tweet = 'Your song is annoying'
print extract_features(tweet.split())
sentiment = classifier.classify(extract_features(tweet.split()))
print sentiment
    
accuracy = nltk.classify.accuracy(classifier, test_set)
print accuracy

