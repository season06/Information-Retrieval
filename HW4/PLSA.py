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
    word_dict = {} # word: count
    all_word_len = 0
    
    for doc_name, doc in doc_dict.items(): # 讀取 key value
        tf_dict[doc_name] = {}
        all_word_len += len(doc.split(' '))
        
        for word in doc.split(' '):        # 將 document 的內容拆成 token
                
            if tf_dict[doc_name].get(word, 0):  # 計算 tf
                tf_dict[doc_name][word] += 1
            else:
                tf_dict[doc_name][word] = 1
                
            if word_dict.get(word, 0):  # 計算 tf
                word_dict[word] += 1
            else:
                word_dict[word] = 1
        
    return tf_dict, word_dict, all_word_len

def initialParameter(doc_len, word_len, K):
    T_w = np.random.random([K, word_len])
    d_T = np.random.random([doc_len, K])
    
    for k in range(0, K):
        normalization = sum(T_w[k, :])
        for i in range(0, word_len):
            T_w[k, i] /= normalization

    for j in range(0, doc_len):
        normalization = sum(d_T[j, :])
        for k in range(0, K):
            d_T[j, k] /= normalization
            
    return T_w, d_T

def EStep(doc_len, word_len, tf_dict, K, T_w, d_T, e_step):
    print('EStep: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    
    for i in range(0, doc_len):
        for j in range(0, word_len):
            
            doc, word = doc_list[i], word_list[j]
            if not tf_dict[doc].get(word, 0):
                continue
                
            denominator = 0
            
            for k in range(0, K):
                e_step[i][j][k] = T_w[k][j] * d_T[i][k]
                denominator += e_step[i][j][k]
                
            if denominator == 0:
                for k in range(0, K):
                    e_step[i][j][k] = 0
            else:
                for k in range(0, K):
                    e_step[i][j][k] /= denominator
    
    return e_step

def MStep(doc_len, word_len, tf_dict, K, T_w, d_T, e_step):
    print('MStep: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    
    ### update T_w : p(wi|Tk)
    for k in range(0, K):
        denominator = 0
        
        for j in range(0, word_len):
            T_w[k][j] = 0
            
            for i in range(0, doc_len):
                doc, word = doc_list[i], word_list[j]
                if tf_dict[doc].get(word, 0):
                    T_w[k][j] += tf_dict[doc][word] * e_step[i][j][k]
            denominator += T_w[k][j]
            
        if denominator == 0:
            for j in word_len:
                T_w[k][j] = 1.0 / word_len
        else:
            for j in range(0, word_len):
                T_w[k][j] /= denominator
                
    ### update d_T : p(Tk|dj)
    for i in range(0, doc_len):
        for k in range(0, K):
            d_T[i][k] = 0
            denominator = 0
            
            for j in range(0, word_len):
                doc, word = doc_list[i], word_list[j]
                if tf_dict[doc].get(word, 0):
                    d_T[i][k] += tf_dict[doc][word] * e_step[i][j][k]
                    denominator += tf_dict[doc][word]
                
            if denominator == 0:
                d_T[i][k] = 1.0 / K
            else:
                d_T[i][k] /= denominator
                
    return T_w, d_T

def Likelihood(doc_len, word_len, tf_dict, K, T_w, d_T):
    likelihood = 0
    
    for i in range(0, doc_len):
        for j in range(0, word_len):
            tmp = 0
            
            for k in range(0, K):
                tmp += T_w[k][j] * d_T[i][k]
                
            if tmp > 0:
                doc, word = doc_list[i], word_list[j]
                if tf_dict[doc].get(word, 0):
                    likelihood += tf_dict[doc][word] * math.log(tmp, 10)
                
    return likelihood

def EM_algorithm(doc_len, word_len, tf_dict, K, T_w, d_T, e_step):
    Iteration = 20
    threshold = 100.0
    oldLoglikelihood = 1
    newLoglikelihood = 1

    for i in range(0, Iteration):
        e_step = EStep(doc_len, word_len, tf_dict, K, T_w, d_T, e_step)
        T_w, d_T = MStep(doc_len, word_len, tf_dict, K, T_w, d_T, e_step)
        newLoglikelihood = Likelihood(doc_len, word_len, tf_dict, K, T_w, d_T)
        
        print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", i+1, " iteration  ", str(newLoglikelihood))
        
        if(oldLoglikelihood != 1 and newLoglikelihood - oldLoglikelihood < threshold):
            break
        oldLoglikelihood = newLoglikelihood
    
    return T_w, d_T

def PLSA_model(query, doc_dict, tf_dict, BG_word, K, T_w, d_T):
    alpha, beta = 0.8, 0.2
    score_dict = {}
    
    for i in range(0, len(doc_list)):
        doc_name = doc_list[i]
        doc = doc_dict[doc_name]
        doc_len = len(doc)
        score = 0
        
        for word in query.split(' '):
            tf = tf_dict[doc_name].get(word, 0)  # 將 word 轉成 score
            
            tmp = 0
            id_word = word2id[word]
            for k in range(0, K):
                tmp += T_w[k][id_word] * d_T[i][k]
            
            first = alpha * (tf / doc_len)
            second = beta * tmp
            third = (1 - alpha - beta) * BG_word[word]
            
            score = first + second + third
            
        score_dict[doc_name] = score
    
    rank = sorted(score_dict.items(), key=lambda x: x[1], reverse = True) # 根據分數做排序
    return rank

### Open file
query_dict, doc_dict = openFile()

### Calculate tf & word count & total word length
tf_dict, word_dict, all_word_len = cal_tf(doc_dict)
print(all_word_len)

### New word dict: select word if word in query or tf > 30
new_word_dict = {}
word2id = {}
i = 0

query_word = []
for _, value in query_dict.items():
    query_word.append(value.split(' '))
query_word = sum(query_word, [])

# select word if word in query or tf > 30
for word in list(word_dict.keys()):
    if word in query_word or word_dict[word] > 40:
        new_word_dict[word] = word_dict[word]

        word2id[word] = i
        i += 1

print(len(new_word_dict))

### Calculate BG word
BG_word = {}
for word, count in new_word_dict.items():
    BG_word[word] = count / all_word_len

### Parameters
doc_list, word_list = list(doc_dict.keys()), list(word_dict.keys())
doc_len, word_len = len(doc_dict), len(new_word_dict)
K = 16  # number of TOPIC

# T_W[topic][word] : p(wi|Tk)
# D_T[doc][topic] : p(Tk|dj)
# e_step[doc][word][topic] : p(Tk|wi,dj)
T_w, d_T = initialParameter(doc_len, word_len, K)
e_step = np.zeros([doc_len,word_len,K])
print(T_w.shape)
print(d_T.shape)
print(e_step.shape)

### EM algorithm
T_w, d_T = EM_algorithm(doc_len, word_len, tf_dict, K, T_w, d_T, e_step)

### Training model & Save answer
f = open('ans.txt', 'w')
string = 'Query,RetrievedDocuments\n'

for _id, _query in query_dict.items():
    rank = PLSA_model(_query, doc_dict, tf_dict, BG_word, K, T_w, d_T)

    string += _id + ','
    for i, doc in enumerate(rank):
        string += doc[0] + ' '
        
        if i == 999:
            break
    string += '\n'
    
f.write(string)
f.close()
print('done')