# -*- coding: utf-8 -*-
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np

############################################Preprocessing the text#################################################################


tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    # create sample documents
doc_a = "I like to eat chocolates."
doc_b = "Father says,chocolates are bad for health"
doc_c = "Sister says candies are good"
doc_d = "I was told by mother to have healthy food"
doc_e = "I love my family,but i love sweets too" 
# compile sample documents into a list
doc_set = [doc_a, doc_b,doc_c,doc_d,doc_e]
# list for tokenized documents in loop
doc_len=len(doc_set)
print(doc_len)
texts = []
j=0
total_words=0
wordsindoc=[]
text=[]
# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    #print(tokens)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    text+=stemmed_tokens
    wordsindoc.append(len(stemmed_tokens))
    j+=1
texts=sorted(set(text))

print(texts)
total_words=len(texts)
print(total_words)
####################################################################################################################################
#############################################INITIALISATION#########################################################################

##################################Calculation of intial ndk,ndw,nk and z############################################################


cust=4									##Custom number of topics(specified by user)
alpha=1
eta=0.001

b=np.random.randint(cust,size=(total_words,doc_len))		##Randomly initialize the array b.b is a word vs document matrix which contains topics assigned to each word in corresponding documents
#print(b)	
z=np.random.randint(cust,size=total_words)
print(z)

##Initialize nk,nkw,ndk with zero arrays.								
nkw=np.zeros((total_words,cust))
ndk=np.zeros((cust,doc_len))
nk=[0]*cust

##Calculate nkw,nkz,nk
#print(total_words)
for k in range(0,total_words):
	for l in range(0,doc_len):
		for x in range(0,cust):
			if b[k][l]==x:
				nk[x]+=1
				nkw[k][x]+=1
	
for m in range(0,doc_len):
	for n in range(0,total_words):
		for y in range(0,cust):
			if b[n][m]==y:
				ndk[y][m]+=1
	

print(nk)
print(nkw)
print(ndk)
wordsintopic=[]

for f in xrange(cust):
	tsum=0
	for g in xrange(total_words):									#Calculate number of words in each topic.					
		tsum+=nkw[g][f]							
	wordsintopic.append(tsum)
#print(wordsintopic)	

toparr=[]
for y in xrange(cust-1):
	toparr.append(y)										#Put topic numbers in an array
#print(toparr)

#print(texts[19])

custiter=500												#Number of iterations specified by user.
tempsum=0
#print(texts[0])
while(custiter!=0):
	for d in xrange(doc_len):
		for w in xrange(total_words):
			word=texts[w]
			topic=z[w]
			#t0=b[w][d]									
			ndk[topic][d]-=1							#Decrement ndk,nkw.
			nkw[w][topic]-=1
			nk[topic]-=1	
			#print(t0,word)
			denom_a=wordsindoc[d]+cust*alpha	#Calculate denominator for p_z
			#print(denom_a,denom_b)
			p_z=[]
			for k in xrange(cust-1):
				#print(k)
				denom_b=wordsintopic[k]+total_words*eta
				#print(wordsintopic[k],denom_a,denom_b)
				x=((nkw[w][k]+eta)/denom_b)*((ndk[k][d]+alpha/denom_a))			#Calculate p_z 
				if x>0:
					p_z.append(x)
				else:
					p_z.append(0)				
				tempsum+=x
			#print(p_z)			
			probs=np.array(p_z/tempsum)
			probs/=probs.sum()
			#print(probs)
			#t0=np.random.choice(toparr,1,p=probs)		#Sampling from multinomial distribution.
			topic=np.random.multinomial(1,probs).argmax()
			#rand=np.random.randint(0,len(t0[0]))						
			#topic=t0[0][rand]
			#print(topic)
			z[w]=topic			
			#b[w][d]=t1									#Increment ndk,nkw.
			ndk[topic][d]+=1
			nkw[w][topic]+=1
			#print(nkw)
			nk[topic]+=1			
			'''if(t0!=t1):
				print('doc:'+str(d),'token:'+str(word),'topic:'+str(t0),'=>',str(t1))
			'''
	#print(custiter)	
	custiter-=1
print(z)
def rowsum(a):
	rsum=0
	rsum=a.sum(axis=1)
	#print(a)
	return(a)

nkw1=np.zeros((total_words,cust))

for i in range(0,cust):
	#print('\n')
	for j in range(0,doc_len):
		theta=(ndk[i][j]+alpha)			
		#print(theta)
for s in range(0,total_words):
	for v in range(0,cust):
		#print(nkw[s][v])
		nkw1[s][v]=(nkw[s][v]+eta)	
#print(nkw)

phi=np.zeros((total_words,cust))

for b in range(0,total_words):
	for c in range(0,cust):
		phi=(nkw1[b][c])/rowsum(nkw1)
#print(phi)
print(texts)



