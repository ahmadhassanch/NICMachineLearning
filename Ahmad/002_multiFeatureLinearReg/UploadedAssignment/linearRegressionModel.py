import numpy as np

def hypothesis(theta, x):
	h = x.dot(theta);
	return h;

def lossFunction(theta, X, y):
	m = X.shape[0];
	h  = hypothesis(theta, X);
	error = 0.5/m*np.sum((h - y)**2)
	return error;