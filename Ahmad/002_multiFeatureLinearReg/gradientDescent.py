import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(hypFunc, lossFunc, i, X, y, alpha, theta):
	m = X.shape[0];
	n = X.shape[1];
	
	hy = hypFunc(theta, X) - y;
	p = hy * X.T;
	dJdx =  np.sum(p,1);
	theta = theta - (alpha / m) * dJdx
	loss = lossFunc(theta, X, y)
	return theta, loss

def gradientDescentLoop(hypFunc, lossFunc, X, y, thetaEst, alpha, nIterations):
	lossArr = [];
	thetas = [];
	for i in range(nIterations):
		thetaEst, loss = gradientDescent(hypFunc, lossFunc, i, X, y, alpha, thetaEst)
		lossArr.append(loss);
		thetas.append(thetaEst);
	return thetas, lossArr

def plotCost(thetas, cost, block):
	plt.subplot(211)
	plt.plot(cost)
	plt.ylabel('Loss');
	
	plt.legend(['Loss Function']);
	plt.grid()

	plt.subplot(212)
	plt.plot(thetas)
	plt.grid()
	nt = thetas[0].shape[0]

	lgnd = [];
	for t in range(nt):
		lgnd.append("theta" + str(t));
	plt.legend(lgnd);
	plt.ylabel('Theta');
	plt.xlabel('Iterations');
	plt.show(block = block)
