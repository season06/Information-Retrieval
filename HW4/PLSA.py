import numpy as np
import math
import random
import time

def openFile():
    query = {}
    doc = {}
    root_path = './data'

    with open(f'{root_path}/query_list.txt', 'r') as file:
        query_list = file.read()
        for file_name in query_list.split('\n'):
            try:
                file_path = f'{root_path}/queries/{file_name}.txt'
                with open(file_path, 'r') as f:
                    query[file_name] = f.read().lower()
            except Exception as e:
                print(e)

    with open(f'{root_path}/doc_list.txt', 'r') as file:
        doc_list = file.read()
        for file_name in doc_list.split('\n'):
            try:
                file_path = f'{root_path}/docs/{file_name}.txt'
                with open(file_path, 'r') as f:
                    doc[file_name] = f.read().lower()
            except Exception as e:
                print(e)

    print(len(query))
    print(len(doc))
    
    return query, doc

def cal_tf(doc_dict):
    tf_dict = {}   # tf[doc][word]
    word_dict = {} # count number of word
    
    for doc_name, doc in doc_dict.items(): # 讀取 key value
        tf_dict[doc_name] = {}
        for word in doc.split(' '):        # 將 document 拆成 word
            if tf_dict[doc_name].get(word, 0):  # 計算 tf
                tf_dict[doc_name][word] += 1
            else:
                tf_dict[doc_name][word] = 1
                word_dict[word] = 1
        
    return tf_dict, word_dict

def initialParameter(doc_list, word_list, K):
    gamma = {} # gamma[K][word]
    delta = {} # delta[doc][K]
    
    for k in range(0, K):
        tmp = [random.random() for _ in range(0, len(word_list))]
        normalize = sum(tmp)
        gamma[k] = {word: tmp[i] / normalize for i, word in enumerate(word_list)}
            
    
    for j, doc in enumerate(doc_list):
        tmp = [random.random() for _ in range(0, K)]
        normalize = sum(tmp)
        delta[doc] = {k: tmp[k] / normalize for k in range(0, K)}
    
    return gamma, delta

def EStep(doc_list, word_list, K, gamma, delta, e_step):
    
    for doc in doc_list:
        e_step[doc] = {}
        for word in word_list:
            e_step[doc][word] = {}
            denominator = 0
            
            for k in range(0, K):
                e_step[doc][word][k] = gamma[k][word] * delta[doc][k]
                denominator += e_step[doc][word][k]
                
            if denominator == 0:
                for k in range(0, K):
                    e_step[doc][word][k] = 0
            else:
                for k in range(0, K):
                    e_step[doc][word][k] /= denominator
    
    return e_step

def MStep(doc_list, word_list, tf_dict, K, gamma, delta, e_step):
    ### update gamma : p(wi|Tk)
    for k in range(0, K):
        denominator = 0
        
        for word in word_list:
            gamma[k][word] = 0
            
            for doc in doc_list:
                gamma[k][word] += tf_dict[doc].get(word, 0) * e_step[doc][word][k]
            denominator += gamma[k][word]
            
        if denominator == 0:
            for word in word_list:
                gamma[k][word] = 1.0 / len(word_list)
        else:
            for word in word_list:
                gamma[k][word] /= denominator
                
    ### update delta : p(Tk|dj)
    for doc in doc_list:
        for k in range(0, K):
            delta[doc][k] = 0
            denominator = 0
            
            for word in word_list:
                delta[doc][k] += tf_dict[doc].get(word, 0) * e_step[doc][word][k]
                denominator += tf_dict[doc].get(word, 0)
                
            if denominator == 0:
                delta[doc][k] = 1.0 / K
            else:
                delta[doc][k] /= denominator
                
    return gamma, delta

def Likelihood(doc_list, word_list, tf_dict, K, gamma, delta):
    likelihood = 0
    
    for doc in doc_list:
        for word in word_list:
            tmp = 0
            
            for k in range(0, K):
                tmp += gamma[k][word] * delta[doc][k]
                
            if tmp > 0:
                likelihood += tf_dict[doc].get(word, 0) * math.log(tmp, 10)
                
    return likelihood

def EM_algorithm(doc_list, word_list, tf_dict, K, gamma, delta):
    iteration = 100
    threshold = 0.001
    oldLikelihood = 1
    newLikelihood = 1
    e_step = {}
    
    for i in range(0, iteration):
        e_step = EStep(doc_list, word_list, K, gamma, delta, e_step)
        gamma, delta = MStep(doc_list, word_list, tf_dict, K, gamma, delta, e_step)
        newLikelihood = Likelihood(doc_list, word_list, tf_dict, K, gamma, delta)
        
        print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", i+1, " iteration  ", str(newLikelihood))
        
        if(oldLikelihood != 1 and newLikelihood - oldLikelihood < threshold):
            break
        oldLikelihood = newLikelihood
    
    return gamma, delta

def PLSA_model(query, doc_dict, tf_dict, K, gamma, delta):
    alpha, beta = 0.6, 0.4
    score_dict = {}
    
    for doc_name, doc in doc_dict.items():
        doc_len = len(doc)
        score = 0
        
        for word in query.split(' '):
            tf = tf_dict[doc_name].get(word, 0)  # 將 word 轉成 score
            
            tmp = 0
            for k in range(0, K):
                tmp += gamma[k][word] * delta[doc_name][k]
            
            first = alpha * (tf / doc_len)
            second = beta * tmp
            third = (1 - alpha - beta) * (tf / doc_len)
            
            score = first + second + third
            
        score_dict[doc_name] = score
    
    rank = sorted(score_dict.items(), key=lambda x: x[1], reverse = True) # 根據分數做排序
    return rank

query_dict, doc_dict = openFile()

tf_dict, word_dict = cal_tf(doc_dict)

## use query word as word_dict
word_dict = {}
for _id, _query in query_dict.items():
    for word in _query.split(' '):
        word_dict[word] = 1

doc_list, word_list = list(doc_dict.keys()), list(word_dict.keys())

# number of TOPIC
K = 64

# gamma[topic][word] : p(wi|Tk)
# delta[doc][topic] : p(Tk|dj)
# e_step[doc][word][topic] : p(Tk|wi,dj)
gamma, delta = initialParameter(doc_list, word_list, K)

gamma, delta = EM_algorithm(doc_list, word_list, tf_dict, K, gamma, delta)

f = open('ans.txt', 'w')
string = 'Query,RetrievedDocuments\n'

for _id, _query in query_dict.items():
    rank = PLSA_model(_query, doc_dict, tf_dict, K, gamma, delta)

    string += _id + ','
    for i, doc in enumerate(rank):
        string += doc[0] + ' '
        
        if i == 999:
            break
    string += '\n'
    
f.write(string)
f.close()
print('done')