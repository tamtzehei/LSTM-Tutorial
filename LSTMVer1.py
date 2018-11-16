# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:07:53 2018

@author: Tze Hei
"""

import pandas
import numpy
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

#Read and plot data
dataset = pandas.read_csv('international-airline-passengers.csv', usecols = [1], engine = 'python', skipfooter = 3)
plt.plot(dataset)
plt.show()

#Convert data into an array of type float 32
num_pass = dataset.values
num_pass= num_pass.astype('float32')

#Normalize data
scaler = MinMaxScaler(feature_range = (0,1))
num_pass = scaler.fit_transform(num_pass)

#Split data into train and test sets
train_length = int(len(dataset) * 0.67) 
train_set = num_pass[0:train_length, :]
test_set = num_pass[train_length + 1:, :]

#Creates two numpy arrays with old_data having original array and new_data having the next element
def create_dataset(dataset, lookback = 1):
    old_data, new_data = [], []
    for i in range(len(dataset) - lookback - 1):
        old_data.append(dataset[i, 0])
        new_data.append(dataset[i + lookback, 0])
    
    return numpy.array(old_data), numpy.array(new_data)

#Create train and test sets
train_old, train_new = create_dataset(train_set)      
test_old, test_new = create_dataset(test_set)

train_old = numpy.reshape(train_old, (train_old.shape[0], 1, train_old.shape[1]))

def create_model():
    model = Sequential()
    
    model
    return model
    
    