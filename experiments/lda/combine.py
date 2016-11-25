import nltk
import string
from nltk.tokenize.punkt import PunktWordTokenizer  

temp_list=
f= open("pass.txt","r",encoding="utf-8-sig")
datas = f.readlines()

for line in datas:
	exclude = set(string.punctuation)
	line = ''.join(ch for ch in line if ch not in exclude)
	print ('\n-----\n'.join(nltk.word_tokenize(line)))
	print ('\n-----\n')

f.close()


