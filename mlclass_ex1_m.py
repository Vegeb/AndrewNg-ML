# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 17:08:25 2015
mat_contents = sio.loadmat('data.mat')
print mat_contents
@author: dbi
"""

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from functions import computeCost, gradientDescent, featureNormalize

# ================ Part 1: Feature Normalization ================

# Load Data
data = pd.read_csv("ex1data2.txt", header=None)
X = data.ix[:,0:1]
y = data.ix[:,2]
m = len(y)

# Scale features and set them to zero mean
X, mu, sigma = featureNormalize(X)

#Add intercept term to X
one = pd.Series(np.ones(m))

X = pd.concat([one,X],axis=1)

# ================ Part 2: Gradient Descent ================
# Choose some alpha value

alpha = 0.01
alpha2 = 0.03
alpha3 = 0.1
alpha4 = 0.3
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
theta2 = np.zeros((3, 1))
theta3 = np.zeros((3, 1))
theta4 = np.zeros((3, 1))
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
theta2, J_history2 = gradientDescent(X, y, theta2, alpha2, num_iters)
theta3, J_history3 = gradientDescent(X, y, theta3, alpha3, num_iters)
theta4, J_history4 = gradientDescent(X, y, theta4, alpha4, num_iters)

# plot convergence graph 
plt.subplot(211)
plt.plot(range(len(J_history)),J_history)
plt.subplot(212)
plt.plot(range(50),J_history2[:50])
plt.plot(range(50),J_history3[:50])
plt.plot(range(50),J_history4[:50])
plt.xlabel('Number of Iterations')
plt.show()