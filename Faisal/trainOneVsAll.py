import numpy as np
import pdb
import scipy.optimize as op
from hypothesis import hypothesis
import lrCostFunction as lrcf


def lrCostFunction(theta, X, y, Lambda):
	m = len(y);
	
	hyp = hypothesis(theta, X) 
	y1  = y.dot(np.log(hyp))
	y0  = (1-y).dot(np.log(1-hyp))
	J   = ((-1.0/m) * (y1+y0)) + Lambda/(2.0*m) * np.sum(theta[1:]**2)

	grad = (1.0/m) * ((hypothesis(theta, X)-y).dot(X))
	temp = theta
	temp[0] = 0
	gradR = grad + ((Lambda/m)*theta)

	return J, gradR 


def trainOneVsAll(X, y, num_labels, Lambda):

	m, n = X.shape
	all_theta = np.zeros((num_labels, n)) # 10x401

	for c in xrange(0, num_labels):
		
		print "Training", c ,"category...", num_labels
		Y = (y==c).astype(int)   # converted True/False to 0/1
		initial_theta = np.zeros((n, 1))
		
		theta = op.minimize(lrCostFunction, x0=initial_theta, args=(X, Y, Lambda), options={'disp': True, 'maxiter':13}, method="Newton-CG", jac=True)
		'''
		theta = op.minimize(lrCostFunction, x0=initial_theta,
											args=(X, Y, Lambda),
											options={'disp': True, 'maxiter':13},
											method="Newton-CG",
											jac=True)
		'''
		all_theta[c,:] = theta["x"]

	return all_theta
	