import pdb
import re
import sys
import nltk
import math
import time
import collections
import pickle
import string
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

TESTING=False
PRACTICAL_WAY=True
pickle_ready=True


NOT_FOUND_MAGIC_NUMBER  = 0.000123

TOTAL_N_WORDS=0
TOTAL_N_SENTENCES=0

notgood=list(string.punctuation)

data=[]
uni_dic={}
bi_dic={}
tri_dic={}

def preProcess(s):
    return  word_tokenize(s.decode('utf-8'))

def do_all_init():
    global NOT_FOUND_MAGIC_NUMBER 
    global TOTAL_N_WORDS
    global TOTAL_N_SENTENCES
    global data
    global uni_dic
    global bi_dic
    global tri_dic
    
    print("init start.")

    if pickle_ready:
    	print("loading...1")
	uni_dic = pickle.load( open( "save.uni_dic", "rb" ) )
    	print("loading...2")
	bi_dic = pickle.load( open( "save.bi_dic", "rb" ) )
    	print("loading...3")
	tri_dic = pickle.load( open( "save.tri_dic", "rb" ) )
    	print("loading...4")
	TOTAL_N_WORDS     = pickle.load( open( "save.TOTAL_N_WORDS", "rb" ) )
	TOTAL_N_SENTENCES = pickle.load( open( "save.TOTAL_N_SENTENCES", "rb" ) )
    	print("init done. ready to answer...")
	return 

    for fn in ["en_US.blogs.txt", "en_US.news.txt","en_US.twitter.txt"]:
         f1 =   open(fn)
         s='x'
         ccc=0
         while s:
                 s  = f1.readline().replace('\r','').replace('\n','')
                 sss = preProcess(s)
                 data.append( sss )
		 if TESTING == True and ccc == 100:
			break
		 ccc += 1

         print(' %s file loaded' % fn)

    for one in data:
         for i in one:
                 TOTAL_N_WORDS += 1
                 if i in uni_dic:
                     uni_dic[i] = uni_dic[i] + 1
                 else:
                     uni_dic[i]=1
    print("uni_dic done")

    for s in data:
         TOTAL_N_SENTENCES += 1
         bi = list( nltk.bigrams(s))
         for i in bi:
            if i in bi_dic:
                bi_dic[i] = bi_dic[i] + 1
            else:
       
                bi_dic[i]=1
    print("bi_dic done")

    for s in data:
         for i in list( nltk.trigrams(s)):
             if i in tri_dic:
                 tri_dic[i] = tri_dic[i] + 1
             else:
                 tri_dic[i]=1 

    print("tri_dic done")

    pickle.dump( uni_dic, open( "save.uni_dic", "wb" ) )
    pickle.dump( bi_dic, open( "save.bi_dic"  , "wb" ) )
    pickle.dump( tri_dic, open( "save.tri_dic", "wb" ) )
    pickle.dump( TOTAL_N_WORDS, open( "save.TOTAL_N_WORDS", "wb" ) )
    pickle.dump( TOTAL_N_SENTENCES, open( "save.TOTAL_N_SENTENCES", "wb" ) )
    print("init done. ready to answer...")

#====================================================================

def unigram_p(i):
    if i in uni_dic:
        return  uni_dic[i] / float(TOTAL_N_WORDS)
    else:
        return NOT_FOUND_MAGIC_NUMBER 

def bigram_p((a,b)):
    if (a,b) in bi_dic:
        if a == '*':
            return  (bi_dic[(a,b)]-0.5) / float(TOTAL_N_SENTENCES)
        else:
            return  (bi_dic[(a,b)]-0.5) / float(uni_dic[a] ) 
    else:
        return NOT_FOUND_MAGIC_NUMBER 

def trigram_p((a,b,c)):
     if a == '*' and b == '*':
         if (a,b,c) in tri_dic:
            return (tri_dic[(a,b,c)]-0.5) / float(TOTAL_N_SENTENCES)
         else: #not found
            return NOT_FOUND_MAGIC_NUMBER 
     else:
         if (a,b) in bi_dic and  (a,b,c) in tri_dic:
            return (tri_dic[(a,b,c)]-0.5) / float(bi_dic[(a,b)]) 
         else: #not found
            return NOT_FOUND_MAGIC_NUMBER 

def qML_bi(v,w):

    if (v,w) in bi_dic:
            return  (bi_dic[(v,w)]-0.5) / float(uni_dic[v] ) 

    A=set([])
    B=set([])

    for i in uni_dic:
        if (v,i) in bi_dic:
            A.add(i)
        else:
            B.add(i)

    summ=0
    for www in A:
        summ += bigram_p((v,www))

    alpha_w_i_minus_1 = 1 - summ

    
    summ=0
    for www in B:
        summ += unigram_p(www)

    ret = alpha_w_i_minus_1 * unigram_p(w) / summ

    return ret


def get_best_word(question):
    question = question.replace('\r','').replace('\n','')

    answer='xx'
    maxval = 0.0
    val=0.0
    A=set([])
    B=set([])

    #########################
    # [u  v]  w
    #########################
    out = preProcess(question)

    #error
    if len(out) < 2:
	return "Error:You must give me more than 2 words"

    # prefix: last 2 word
    (u,v) = preProcess(question)[-2:]

    for i in uni_dic:
        if (u,v,i) in tri_dic:
            A.add(i)
        else:
            B.add(i)

    summ=0
    for guess in A:
        val = trigram_p((u,v,guess))
        #print( "%20s : %f" % (guess,val) )
        assert( val != NOT_FOUND_MAGIC_NUMBER )
        assert( val > 0 and val <= 1)
        if val > maxval and guess not in notgood:
            maxval = val
            answer = guess

        summ += val

    if PRACTICAL_WAY == True:
	return answer
    else:
	#it took too much time!
	alpha = 1 - summ
        summ=0
	for guess in B:
	    summ += qML_bi(v,guess)
	for guess in B:
	    val = alpha* qML_bi(v,guess)/summ
	    assert( val != NOT_FOUND_MAGIC_NUMBER )
	    assert( val > 0 and val <= 1)
	    if val > maxval and guess not in notgood:
	        maxval = val
	        answer = guess

        return answer

#################################################################
###just for test
#################################################################
### #print('--------------------')
### #print(qML_bi('the','book'))
### #print('--------------------')
### #print(qML_bi('the','house'))
### #print(qML_bi('the','buy'))
### #print(qML_bi('the','EOS'))
### #print(qML_bi('the','paint'))
### #print(qML_bi('the','sell'))
### #print(qML_bi('the','the'))
### #print('--------------------')
### 
### print(get_best_word("sell the"))
### print("[oracle!]")
### print('book : 0.500000')
### print('sell : 0.005952')
### print('buy : 0.035714')
### print('house : 0.357143')
### print('EOS : 0.047619')
### print('paint : 0.005952')
### print('the : 0.047619')
### 
################################################################
####just for testing......
###while True:
###        question = raw_input('your input:')
###        if question == 'q':
###            break
###
###        print("input:",question)
###    
###
###        answer = get_best_word( question)
###
###        print('answer:%s' % (answer))
################################################################

from socket import *

print("start..")
do_all_init()

sock  = socket(AF_INET,SOCK_DGRAM)
sock.setsockopt(SOL_SOCKET,SO_REUSEADDR,1)
sock.bind(("127.0.0.1",2000))

while(True):
	string,addr = sock.recvfrom(1024)
	best_answer = get_best_word(string)
	sock.sendto(best_answer,addr)

sock.close()
