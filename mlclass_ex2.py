# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 15:08:15 2015

@author: dbi
"""


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions_2 import plotData, costFunction, plotLinearDecisionBoundary

# ==================== Part 1: Plotting ====================
os.chdir("C:\Continuum\Anaconda\Machine Learning\mlclass-ex2-006") # Change directory
data = pd.read_csv("ex2data1.txt", header=None)
X = data.ix[:,:1]
y = data.ix[:,2]
fig1= plt.figure()
plotData(X,y)
plt.ylabel('Exam 2 Score'); # Set the yaxis label
plt.xlabel('Exam 1 Score'); # Set the xaxis label
plt.legend(['Admitted', 'Rejected'])
fig1.show()

# ============ Part 2: Compute Cost and Gradient ============
m, n = X.shape
X_t = pd.concat([pd.Series(np.ones(m)),X],axis=1) 
# convert into numpy matrix
X_t = np.mat(X_t.as_matrix())
y_t = y.as_matrix()[:,np.newaxis]

#initial_theta = np.zeros([n+1,1])
initial_theta = np.zeros(n+1)

cost0, grad= costFunction(initial_theta, X_t, y_t)

print 'Cost at initial theta (zeros): %.2f' % cost0
print 'Gradient at initial theta (zeros): ' 
print grad

# ============= Part 3: Optimizing using fminunc  =============
#result_BFGS = minimize(lambda t: costFunction(t,X,y), initial_theta, method='BFGS', jac=True)

result_Newton = minimize(lambda t: costFunction(t,X_t,y_t), initial_theta, method='Newton-CG', jac=True)
print result_Newton

cost = result_Newton['fun']
theta = result_Newton['x']
# Plot Boundary
fig2= plt.figure()
plotLinearDecisionBoundary(theta, X, y);
fig2.show()
