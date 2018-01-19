import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plotCost3D(thetas, cost, X, y, block):
	fig = plt.figure()
	gs = plt.GridSpec(8, 1)

	ax1 = fig.add_subplot(gs[:2,0])
	ax1.plot(cost)
	plt.ylabel('Loss');
	plt.legend(['Loss Function']);
	plt.grid()

	ax2 = fig.add_subplot(gs[3:5, 0])
	ax2.plot(thetas)
	plt.grid()
	nt = thetas[0].shape[0]

	lgnd = [];
	for t in range(nt):
		lgnd.append("theta" + str(t));
	plt.legend(lgnd);
	plt.ylabel('Theta');
	plt.xlabel('Iterations');

	ax3 = fig.add_subplot(gs[6:,0], projection='3d')
	ax3.scatter(X[:, 1], X[:, 2], y, c=y, marker='o')
	ax3.set_xlabel('X1 Feature X')
	ax3.set_ylabel('X2 Feature Y')
	ax3.set_zlabel('Y  labels Z')
	# plt.plot(x1, x2, h)
	plt.show(block=block)

