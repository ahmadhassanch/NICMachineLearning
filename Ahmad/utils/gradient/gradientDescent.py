import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def gradientDescent(hypFunc, lossFunc, i, X, y, alpha, theta):
	m = X.shape[0];
	n = X.shape[1];
	
	hy = hypFunc(theta, X) - y;
	p = hy * X.T;
	dJdx =  np.sum(p,1);
	theta = theta - (alpha *1.0 / m) * dJdx
	#if(np.max(np.abs(theta))>):
	#theta = theta / sum(theta)
	#theta = theta / theta[2]
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

def plotThetas(thetas, ylabel):
	plt.plot(thetas)
	plt.grid()
	nt = thetas[0].shape[0]

	lgnd = [];
	for t in range(nt):
		lgnd.append("theta" + str(t));
	plt.legend(lgnd);
	plt.ylabel(ylabel);
	plt.xlabel('Iterations');

def plotCost(thetas1, cost, block):
	thetas = np.copy(thetas1)	
	print " thetas", thetas.shape
	#	maxVal = np.max(thetas[-1]);
	for i in range(thetas.shape[0]):
		thetas[i,:] = thetas[i,:] / thetas[i,0]
	#thetas = thetas / thetas[0]
	#print thetas
	plt.subplot(311)
	plt.plot(cost)
	plt.ylabel('Loss');
	
	plt.legend(['Loss Function']);
	plt.grid()

	plt.subplot(312)
	plotThetas(thetas1, 'theta')
	plt.subplot(313)
	plotThetas(thetas, 'normalized theta')
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
	ax3.set_xlabel('X1')
	ax3.set_ylabel('X2')
	ax3.set_zlabel('Y')
	# plt.plot(x1, x2, h)
	plt.show(block=block)

def plotDataAndDecisionBoundary3D(X, thetaFinal, y, block):
	x1 = X[:,1];
	x2 = X[:,2];
	x3 = X[:,3];
	
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(x1,x2,x3,c=y,marker='o')
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('x3')

	x1min = np.min(x1);
	x1max = np.max(x1);
	x2min = np.min(x2);
	x2max = np.max(x2);
	x1 = np.array([x1min,x1max,x1max,x1min])
	x2 = np.array([x2min,x2min, x2max, x2max])
	#x3 = np.array([.5, .5, .5, .5])

	t0 = thetaFinal[0]/thetaFinal[3]
	t1 = thetaFinal[1]/thetaFinal[3]
	t2 = thetaFinal[2]/thetaFinal[3]
	x3 = -t0 - t1*x1 - t2*x2; 
	verts = [zip(x1, x2, x3)]
	ax.add_collection3d(Poly3DCollection(verts))
	plt.show(block = block)