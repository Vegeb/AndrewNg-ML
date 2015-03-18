# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:23:01 2015

@author: dbi
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = pd.read_csv('ex2data1.txt', header=None)
X1 = data[[0, 1]]
y1 = data[2]

def mapFeature(X, degree=1):
    """
    Take X, an m x 2 DataFrame, and return a numpy array with more features,
    including all degrees up to degree.
    1, X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, etc.
    """

    m, n = np.shape(X)

    if not n == 2:
        raise ValueError('mapFeature supports input feature vectors of length 2, not %i' % n)

    out = np.ones([1, m])

    for totalPower in xrange(1, degree+1):
        for x1Power in xrange(0, totalPower+1):
            out = np.append(out, [X[0]**(totalPower-x1Power) * X[1]**x1Power], axis=0)

    return out.T

X_array = mapFeature(X1)
y_array = np.array(y1)
m, n = np.shape(X1)

def sigmoid(z):
    """
    Compute sigmoid functoon
    Compute the sigmoid of each value of z (z can be a matrix, vector, or scalar).
    Accepts a scalar object, numpy array, Series, or DataFrame.
    """

    g = 1 / (1 + np.exp(-1*z))

    return g
    
def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression
    COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
    """

    m = len(y) * 1.0

    cost = 1/m * (
        np.dot(-1*y, np.log(sigmoid(np.dot(X, theta))))
        - np.dot(1 - y, np.log(1 - sigmoid(np.dot(X, theta))))
        )

    grad = 1/m * np.dot(sigmoid(np.dot(X, theta)) - y, X)

    return cost, grad

initial_theta1 = np.zeros(n + 1)
cost1, grad1 = costFunction(initial_theta1, X_array, y_array)

result_BFGS = minimize(lambda t: costFunction(t, X_array, y_array),
                           initial_theta1, method='BFGS', jac=True)