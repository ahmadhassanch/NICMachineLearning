import numpy as np

def sigmoid(z):
    gx = 1/(1+np.exp(-z))
    return gx

def hypothesis(theta, x):
    z = x.dot(theta);
    h = sigmoid(z);
    return h

def lossFunction(theta, X, y):
    m = X.shape[0];
    h  = hypothesis(theta, X);
    p1 = -y*np.log(h)
    p2 = (1-y)* np.log(1.0-h)
    J = np.mean(p1 - p2)
    return J;
