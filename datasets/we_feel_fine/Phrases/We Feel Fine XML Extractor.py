
import urllib.request
import urllib
import re
import time
from xml.dom import minidom

files = ["HAPPY.txt","ANGRY.txt","DISGUSTED.txt","SAD.txt","FEAR.txt","SURPRISED.txt"]
files1 = ["joyTest.txt","angerTest.txt","disgustTest.txt","sadnessTest.txt","fearTest.txt","surpriseTest.txt"]


for fi,fs in zip(files1,files):
       #try:
            html= open(fi,"rU")
            f1= open(fs,"w")
            doc = minidom.parse(html)
            html.close()
            sentences = doc.getElementsByTagName("feeling")
            for sentence in sentences:
                  f1.write(sentence.getAttribute("sentence")+"\n")
            f1.flush()
            f1.close()
       #except :
            #print ("Nope")

print("Success!!")
        
