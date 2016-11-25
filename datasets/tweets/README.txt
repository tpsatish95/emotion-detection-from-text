=======

Introduction

This dataset was collected using Twitter public streaming API between Nov. 10th, 2011 and Dec. 22nd 2011. The collected tweets were automatically labeled using the emotion hashtags at the end of each tweet. Please refer to our paper (listed below) for more details about collecting and labeling these tweets.

Aug 9 2012

=======

Contact

Wenbo Wang
Ohio Center for Excellence in Knowledge-enabled Computing (Kno.e.sis Center)
Wright State University
Email: wenbo@knoesis.org
Homepage: http://knoesis.wright.edu/students/wenbo/

Lu Chen
Ohio Center for Excellence in Knowledge-enabled Computing (Kno.e.sis Center)
Wright State University
Email: chen@knoesis.org
Homepage: http://knoesis.wright.edu/students/luchen/
    
=======

Citation Info

Wenbo Wang, Lu Chen, Krishnaprasad Thirunarayan, Amit P. Sheth. Harnessing Twitter ‘Big Data’ for Automatic Emotion Identification. IEEE fourth conference on social computing, 2012
http://knoesis.org/library/resource.php?id=1749

=======

List of files 

Data Format Summary 
The tweets are automatically labeled with the following seven emotions: anger, fear, joy, love, sadness, surprise and thankfulness. Each of the file has the following format:
tweetId	<tab> emotionLabel

In feature combination experiment, we used train_1.txt (248,898) for training and test.txt (250,000) for testing.
In examing the effects of increasing the sizes of training data, we used train_2_1.txt (1,000), train_2_2.txt (10,000), … , train_2_10.txt for training and the same test.txt (250,000) for testing.
Apart from the above-mentioned files, dev.txt was used as a development dataset.

=======

FAQ

Q: Why do you share Twitter id only? Why not share Twitter text too?
A: Twitter has made very specific restrictions on redistributing Twitter data (Refer to 1.4.a in https://dev.twitter.com/terms/api-terms). We'd love to share Twitter text, but this is forbidden by Twitter. And the best we can do is to share Twitter ids. Similar cases happened in the past: 
"Twitter Asks DiscoverText to Stop Sharing Tweet Data"
http://blog.texifter.com/index.php/2011/05/04/twitter-cites-terms-of-service-violation/

"As per request from Twitter the data is no longer available."
http://snap.stanford.edu/data/twitter7.html

Q: How can I get Twitter text by Twitter id?
A: Twitter has a special API (https://dev.twitter.com/docs/api/1/get/statuses/show/%3Aid), which returns a tweet based on the Twitter id.    We recommend you to use Twitter4j (http://twitter4j.org/) API to collect tweets (https://groups.google.com/forum/?fromgroups#!topic/twitter4j/oJhFpK4CDsE%5B1-25%5D).

=======