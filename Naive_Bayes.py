# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:49:32 2020

@author: Ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize


ps = PorterStemmer()
file = open("a1_d3.txt",'r')
data=file.read()

temp_dataset=data.split("\n")
X=[]
Y=[]
no_of_ones=0
tot=0
for i in range(len(temp_dataset)-1):
    temp=temp_dataset[i].split("\t")
    temp[0]=temp[0].lower()
    #temp[0]=re.sub(r'\d+', '', temp[0]) #remove numbers
    #temp[0]=temp[0].translate(maketrans("",""), string.punctuation) #remove punctuations
    #temp[0] = temp[0].strip() #remove \t
    X.append(temp[0])
    Y.append(int(temp[1]))
    tot+=1



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#------------------------------------------------------------------------
no_of_ones=0
no_of_zeros=0
final_dict={}
tot=len(X_train)
for i in range(len(X_train)):
    if(Y_train[i]==1):
        no_of_ones+=1
    else:
        no_of_zeros+=1
    words = word_tokenize(X_train[i]) #tokenized words
    stemmed_words = []
    for w in words:
        stemmed_words.append(ps.stem(w))
    temp=np.unique(stemmed_words)
    for j in range(len(temp)):
        #if(temp[i].isalpha()):
        if temp[j] in final_dict.keys():
            if(Y_train[i]==1):
                final_dict[temp[j]][0]+=1
            else:
                final_dict[temp[j]][1]+=1
        else:
            final_dict[temp[j]]=[0,0]
            if(Y_train[i]==1):
                final_dict[temp[j]][0]+=1
            else:
                final_dict[temp[j]][1]+=1
                
#-----------------------------------------------------------------------------
#test
predict=np.zeros((1,len(Y_test))).astype(int)
for i in range(len(X_test)):
    words = word_tokenize(X_test[i])
    stemmed_words = []
    for w in words:
        stemmed_words.append(ps.stem(w))
    temp=np.unique(stemmed_words)
    prob1=0.5
    prob2=0.5
    for j in range(temp.shape[0]):
        if temp[j] in final_dict.keys():
            prob1*=final_dict[temp[j]][0]/(final_dict[temp[j]][1]+final_dict[temp[j]][0])
            prob2*=final_dict[temp[j]][1]/(final_dict[temp[j]][1]+final_dict[temp[j]][0])
        else:
            prob1*=0.5
            prob2*=0.5
            
    #print(f"{i} = {prob1} +{prob2}")
    prob1*=(no_of_ones)/len(X_train)
    prob2*=(no_of_zeros)/len(X_train)
    if(prob2>prob1):
        predict[0][i]=0
    else:
        predict[0][i]=1

Y_temp = np.array(Y_test).reshape(1,len(Y_test))        
check=np.equal(Y_test,predict)
unique, counts = np.unique(check, return_counts=True)
temp=dict(zip(unique, counts))
test_accuracy=temp[True]/Y_temp.shape[1]
print(test_accuracy)
        
                
        
                

