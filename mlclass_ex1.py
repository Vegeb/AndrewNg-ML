# -*- coding: utf-8 -*-
"""
Created on Tue Jan 06 14:16:38 2015

@author: dbi
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from functions import computeCost, gradientDescent

# ======================= Part 2: Plotting =======================
print ("Plotting Data ...\n")

#os.chdir("C:\Users\dbi\Documents\Python Scripts\Machine Learning\mlclass-ex1-006\mlclass-ex1") # Change directory
data = pd.read_csv("ex1data1.txt", header=None)
X = data.ix[:,0]
y = data.ix[:,1]
m = len(y) # No. of training examples
plt.plot(X,y,'rx',markersize=10)# Plot Data
plt.ylabel('Profit in $10,000s'); # Set the yaxis label
plt.xlabel('Population of City in 10,000s'); # Set the xaxis label


# =================== Part 3: Gradient descent ===================
print ('Running Gradient Descent ...\n')

one = pd.Series(np.ones(m))

X = pd.concat([one,X],axis=1) #Add a column of ones to x
theta = pd.DataFrame(np.zeros((2,1)))

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print ('Theta found by gradient descent:')
print (theta)
plt.plot(X[1],X.dot(theta))
plt.legend(['Training data', 'Linear regression'])
plt.show()

#============= Part 4: Visualizing J(theta_0, theta_1) =============
print ('Visualizing J(theta_0, theta_1) ...\n')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
   # fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [theta0_vals[i], theta1_vals[j]] 
        J_vals[i][j] = computeCost(X,y,t)
        
J_vals = pd.DataFrame(J_vals)
J_vals = J_vals.transpose()
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=10, cstride=10, cmap=cm.coolwarm)
cset = ax.contour(theta0_vals, theta1_vals, J_vals, zdir='z', stride=1, offset=-100, cmap=cm.coolwarm)

ax.set_xlabel('theta0')
ax.set_xlim(-10, 10)
ax.set_ylabel('theta1')
ax.set_ylim(-1, 4)
ax.set_zlabel('J_vals')
ax.set_zlim(0, 800)

plt.show()


