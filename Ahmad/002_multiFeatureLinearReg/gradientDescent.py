import numpy as np
import linearRegressionModel as linRegModel

def gradientDescent(i, X, y, alpha, theta):
	m = X.shape[0];
	n = X.shape[1];
	
	hy = linRegModel.hypothesis(theta, X) - y;
	p = hy * X.T;
	dJdx =  np.sum(p,1);
	theta = theta - (alpha / m) * dJdx
	loss = linRegModel.lossFunction(theta, X, y)
	print i, theta, " == ", loss
	return theta, loss



