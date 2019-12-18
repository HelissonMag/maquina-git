# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:24:44 2019

@author: homcerqueira
"""
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    return unique_list
        

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],5) 
        self.y          = y
        self.weights2   = np.random.rand(5,3)     
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

        

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

print('sss')

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    onehotencoder = OneHotEncoder()
    y = iris.target
    aux = []
    for i in y:
        aux.append([i])
    y = aux
    y_one = onehotencoder.fit_transform(y).toarray()

    nn = NeuralNetwork(X,y_one)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()
    print(nn.output)
