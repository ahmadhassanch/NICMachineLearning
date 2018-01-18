import numpy as np
import lossAndHypothesis 

def gradientDescent(i, X, y, alpha, theta):
	m = X.shape[0];
	n = X.shape[1];
	
	hy = lossAndHypothesis.hypothesis(theta, X) - y;
	p = hy * X.T;
	dJdx =  np.sum(p,1);
	theta = theta - (alpha / m) * dJdx
	loss = lossAndHypothesis.lossFunction(theta, X, y)
	print i, theta, " == ", loss
	return theta, loss



