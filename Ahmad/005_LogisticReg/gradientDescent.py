import numpy as np
import logisticRegression as logRegr

def gradientDescent(i, X, y, alpha, theta):
	m = X.shape[0];
	n = X.shape[1];
	
	hy = logRegr.hypothesis(theta, X) - y;
	p = hy * X.T;
	dJdx =  np.sum(p,1);
	theta = theta - (alpha / m) * dJdx
	loss = logRegr.lossFunction(theta, X, y)
	print i, theta, " == ", loss
	return theta, loss



