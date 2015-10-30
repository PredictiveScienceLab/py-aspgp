#!/usr/bin/env python


"""
In this first demo, we demonstrate the capability
of the gradient-free active subspace method by
taking a dataset whose input has 10 features(columns)
and the underlying active subspace is 1 dimensional.

AUTHOR - Rohit K. Tripathy
"""

import GPy
from aspgp import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
np.random.seed(42353)
from pdb import set_trace as keyboard
sns.set_context('paper')

#First we create a synthetic data-set by sampling from a standard normal
#100 samples of the input
X = np.random.randn(100, 10) 

#The projection matrix.
#This will be the true underlying W matrix which our algorithm will hopefully discover
W = np.random.randn(10, 1)

#coefficients of the linear subspace
a = np.random.randn()
b = np.random.randn(1, 1)

#Projected inputs
Z = np.dot(X, W)

#Ouputs
Y = a*np.ones((100, 1)) + np.dot(Z, b)

#define an inner kernel
inner = GPy.kern.RBF(1)

#print the inner kernel
print "=" * 21
print "The inner kernel is :"
print "=" * 21
print inner

#define the active subspace kernel
kernel = ActiveSubspaceKernel(input_dim = 10, inner_kernel = inner)

#print the active subspace kernel
print "=" * 31
print "The active subspace kernel is :"
print "=" * 31
print kernel

#define the active subspace regression model
model = ActiveSubspaceGPRegression(X=X, Y=Y, inner_kernel = inner)

#print the active subspace model
print "=" * 43
print "The active subspace model(unoptimized) is :"
print "=" * 43
print model

#now we optimize the model and save the surrogate if we haven't already done it
if os.path.exists('demo1.pcl'):
    with open('demo1.pcl', 'rb') as f:
        model = pickle.load(f)
else:    
    model.optimize()
    with open('demo1.pcl', 'wb') as f:
        pickle.dump(model, f)
    
#print the optimized active subspace model
print "=" * 41
print "The active subspace model(optimized) is :"
print "=" * 41
print model

#===========================#
#       PREDICTION
#===========================#

#First we generate some test input points.(In this case 10 input points)
X_test = np.random.randn(10, 10)

#get the optimized projection matrix from the surrogate model object
W_opt = -np.array(model.kern.W)

#get the low dimensional projection
Z_test = np.dot(X_test, W_opt)

#Now we predict
Y_predict, C_predict = model.predict(X_test, full_cov = True)

#Now we plot the active subspace predictions 
plt.plot(Z_test, Y_predict, 'bo', label = 'Predictions')

#plot the actual low dimensional model
plt.plot(Z, Y, 'r-', label = 'True model')
plt.legend(loc = 'best')
plt.xlabel('$z=W^{T}x$')
plt.ylabel('$y$')
plt.savefig('demo1_predictions.pdf')










