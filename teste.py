# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:05:13 2019

@author: homcerqueira
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import nnw_scratch_3hiddenlayers as ns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
np.random.seed(2)
#dados = datasets.load_wine()
file = 'train'
dataset = pd.read_csv(file+'.csv') 
X = dataset.iloc[:,:dataset.shape[1]-1].values
y = dataset.iloc[:,dataset.shape[1]-1:dataset.shape[1]+1].values
# The 4 training examples by columns
#---X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
#X = dados.data
#X = dados.data
X = sc.fit_transform(X)
# The outputs of the XOR for every example in X
#------Y = np.array([[0, 1, 1, 0]])
#y = dados.target

onehotencoder = OneHotEncoder(categories='auto')
#aux = []
#for i in y:
#    aux.append([i])
#y = aux
y = onehotencoder.fit_transform(y).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# No. of training examples
m = X.T.shape[1]

# Set the hyperparameters
n_x = X.shape[1]     #No. of neurons in first layer
n_h1 = 6   #No. of neurons in hidden layer 16
n_h2 = 5  #No. of neurons in hidden layer 15
n_h3 = 4  #No. of neurons in hidden layer 12
n_y = y.shape[1]    #No. of neurons in output layer
num_of_iters = 1000
learning_rate = 0.6 #1.2
print('Numero Neuronio Layer 1: '+ str(n_h1))
print('Numero Neuronio Layer 2: '+ str(n_h2))
print('Numero Neuronio Layer 3: '+ str(n_h3))
print('Learning rate:'+ str(learning_rate)) 

trained_parameters = ns. model(X_train.T,y_train.T, n_x, n_h1,n_h2,n_h3, n_y, num_of_iters, learning_rate)

y_predict = np.array(ns.predict(X_test.T, trained_parameters, n_y))
valor = accuracy_score(y_test, y_predict)