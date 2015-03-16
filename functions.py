# -*- coding: utf-8 -*-
"""
Created on Tue Jan 06 16:13:33 2015

@author: dbi
"""
import numpy as np

def computeCost(X,y,theta):
    m = len(y)
    return (0.5/m)*((X.dot(theta).subtract(y,axis='index'))**2).sum()
    
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    
    """
    for i in range(num_iters):
        """
        temp1 = theta.ix[0] - alpha/m*(X.dot(theta).subtract(y,axis='index')).multiply(X[0],axis='index').sum()
        temp2 = theta.ix[1] - alpha/m*(X.dot(theta).subtract(y,axis='index')).multiply(X[1],axis='index').sum()
        theta.ix[0] = temp1
        theta.ix[1] = temp2
        """
        theta = theta - alpha/m*(X.dot(theta).subtract(y,axis='index')).transpose().dot(X).transpose()
        J_history[i] = computeCost(X, y, theta)
    return (theta, J_history)
    
def featureNormalize(X):
    mu = X.mean()
    sigma = X.std()
    X_norm = (X-mu)/sigma
    return (X_norm, mu, sigma)
