# -*- coding: utf-8 -*-
"""
Created on Tue Jan 06 16:13:33 2015

@author: dbi
"""
import numpy as np
from numpy import newaxis, r_, c_, mat, e
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def plotData(X,y):
    plt.plot(X[y==1][0],X[y==1][1],'k+',markersize=7)
    plt.plot(X[y==0][0],X[y==0][1],'ko',markerfacecolor='y',markersize=7)
    
def sigmoid(z):
    return 1.0/(1+np.exp(-z))
    
def costFunction(theta, X, y):
    m = len(y)
    theta = c_[theta]
    J = (1.0/m)*(-y.T*np.log(sigmoid(X*theta))-(1-y).T*np.log(1-sigmoid(X*theta)))
    grad = (1.0/m)*X.T*(sigmoid(X*theta)-y)
    return J.A.flatten(), grad.A.flatten()


def plotLinearDecisionBoundary(theta, X, y):
    plotData(X,y)
    pt_1 = [min(X.ix[:,1])-2, max(X.ix[:,1])+2]
    pt_2 = -1.0/theta[2]*np.add(np.multiply(pt_1,theta[1]),theta[0])
    plt.plot(pt_1,pt_2)

    
def featureNormalize(X):
    mu = X.mean()
    sigma = X.std()
    X_norm = (X-mu)/sigma
    return (X_norm, mu, sigma)
