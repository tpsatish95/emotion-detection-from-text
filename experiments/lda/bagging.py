import nltk
import string
import itertools
import numpy
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
V= len(main_list)
K=15
f.close()


# docs : documents which consists of word array
# K : number of topics
# V : vocaburary size
alpha=1/3
beta=1/3

z_m_n = [] # topics of words of documents
n_m_z = numpy.zeros((len(self.main_list), K)) + alpha     # word count of each document and topic
n_z_t = numpy.zeros((K, V)) + beta # word count of each topic and vocabulary
n_z = numpy.zeros(K) + V * beta    # word count of each topic

for m, doc in enumerate(main_list):
    z_n = []
    for t in doc:
        # draw from the posterior
        p_z = n_z_t[:, t] * n_m_z[m] / n_z
        z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

        z_n.append(z)
        n_m_z[m, z] += 1
        n_z_t[z, t] += 1
        n_z[z] += 1
    z_m_n.append(numpy.array(z_n))




    for m, doc in enumerate(main_list):
    	for n, t in enumerate(doc):
        # discount for n-th word t with topic z
	        z = z_m_n[m][n]
	        n_m_z[m, z] -= 1
	        n_z_t[z, t] -= 1
	        n_z[z] -= 1

	        # sampling new topic
	        p_z = n_z_t[:, t] * n_m_z[m] / n_z # Here is only changed.
	        new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

	        # preserve the new topic and increase the counters
	        z_m_n[m][n] = new_z
	        n_m_z[m, new_z] += 1
	        n_z_t[new_z, t] += 1
	        n_z[new_z] += 1