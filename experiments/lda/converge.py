import nltk
import string
import itertools
from nltk.tokenize.punkt import PunktWordTokenizer  

temp_list=[]
global_list=[]
main_list=[]

f= open("pass.txt","r",encoding="utf-8-sig")
datas = f.readlines()

for line in datas:
	exclude = set(string.punctuation)
	line = ''.join(ch for ch in line if ch not in exclude)
	temp_list=line.split()
	global_list.append(temp_list)
	#print list(itertools.chain(*a))
	#for word in temp_list:
		#global_list.append(word)
	
main_list=list(itertools.chain(*global_list))
print (main_list)
print (len(main_list))

	
f.close()


