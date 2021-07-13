import os
import random
import math
import numpy as np
import scipy as sp
import sklearn
from sklearn.metrics import *
import matplotlib.pyplot as plt

in_size = 785
h_size = 20
out_size = 10
train_set = 60000
test_set = 10000
max_epochs = 50
lrate = 0.1
bias = 1

def sigmoid(x):
    return sp.special.expit(x)
    

def fwd_propih(data):		
    ho = np.matmul(np.reshape(data, (1, in_size)), wi2h) # Compute output array
    return sigmoid(ho+bias)

def fwd_propho(hidden):
    ao = np.matmul(np.reshape(hidden, (1, h_size)), wh2o)
    return sigmoid(ao+bias)

def evaluate(ao, y_t):
    ym = np.insert(np.zeros((1, out_size-1)), np.argmax(ao), 1)				# Compute array representation of predicted output
    t_k = np.insert(np.zeros((1, out_size-1)), y_t, 1)					# Compute array for target value 
    return t_k-ym

def back_propoh(error, ao, ho, wh2o):
    h2og = ao*(1-ao)
    deltak = np.multiply(error, h2og)
    deltakw = np.matmul(ho.T, deltak)
    wh2o -= (lrate * deltakw)									# Compute new set of weights
    return wh2o

def back_prophi(error, ih, ho, ao, wi2h, wh2o):  
    hd = ho*(1-ho)
    h2og = ao*(1-ao)
    deltak = np.multiply(error, h2og)
    herr = np.matmul(deltak, wh2o.T)
    deltaj = np.multiply(herr ,hd)
    deltajw = np.matmul(np.reshape(ih.T, (in_size, 1)), deltaj)
    
    wi2h -= lrate * deltajw                                                                    # Compute new set of weights
    return wi2h


def test_PN(dataset, data_labels, set_size):
    pred = []
    for i in range(0, set_size):
        ho = fwd_propih(dataset[i, :])				# Feed-forward an image sample to get output array
        for i in range(0, h_size):
            ao = fwd_propho(ho)
        pred.append(np.argmax(ao))				# Append the predicted output to pred list 
    return accuracy_score(data_labels, pred), pred

def load_data(file_name):
    data_file = np.loadtxt(file_name, delimiter=',', skiprows=1)
    dataset = np.insert(data_file[:, np.arange(1, in_size)]/255, 0, 1, axis=1)
    data_labels = data_file[:, 0]
    return dataset, data_labels	

# Load Training and Test Sets :
print("\nLoading Training Set")
train_data, train_labels = load_data('mnist_train.csv')
print("\nLoading Test Set\n")
test_data, test_labels = load_data('mnist_test.csv')

arr_epoch = []
arr_train_acc = []
arr_test_acc = []


# Randomize Weights :
wi2h = (np.random.rand(in_size, h_size) - 0.5)*(0.1)				# Generate weight matrix with random weights 						
wh2o = (np.random.rand(h_size, out_size) - 0.5)*(0.1)

# Run Epochs :
epoch = 0
arr_epoch = []
arr_test_acc = []
arr_train_acc = []
while (1):
        curr_accu, pred = test_PN(train_data, train_labels, train_set)			# Test network on training set and get training accuracy
        print("Epoch " + str(epoch) + " :\tTraining Set Accuracy = " + str(curr_accu))
        if epoch==max_epochs:
            break									# If network is converged, stop training
        test_accu, pred = test_PN(test_data, test_labels, test_set)			# Test network on test set and get accuracy on test set
        print("\t\tTest Set Accuracy = " + str(test_accu))
        prev_accu = curr_accu
        epoch+=1
        for i in range(0, train_set):                                  #training cycle
            ho = fwd_propih(train_data[i, :])                               # Feed-forward an image sample to get output array
            ao = fwd_propho(ho)
            error = evaluate(ao, int(train_labels[i]))              # Evaluate to find array representation of predicted output
            wi2h = back_prophi(error, train_data[i, :], ho, ao, wi2h, wh2o)
            wh2o = back_propoh(error, ao, ho, wh2o)           # Back propagate error through the network to get adjusted weights


        arr_epoch.append(epoch) 
        arr_train_acc.append(curr_accu)
        arr_test_acc.append(test_accu)	

# Test Network again :
test_accu, pred = test_PN(test_data, test_labels, test_set)				# Test network on test set and get test accuracy

# Confusion Matrix	
print("\t\tTest Set Accuracy = " + str(test_accu) + "\n\nLearning Rate = " + str(eta) + "\n\nConfusion Matrix :\n")
print(confusion_matrix(test_labels, pred))
print("\n")

# Graph Plot
plt.title("Learning Rate %r" %lrate)
plt.plot(arr_train_acc)
plt.plot(arr_test_acc)
plt.ylabel("Acuracy %")
plt.xlabel("Epochs")
plt.show()

