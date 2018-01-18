import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lossAndHypothesis 

def generateDataPoly(xmin, xmax, m):
	x = np.linspace(xmin, xmax, m);
	np.random.shuffle(x)
	return x


def generateX(xRange, m):
	X = [];
	N = xRange.shape[0];
	for i in range(N):
		x = generateDataPoly(xRange[i,0], xRange[i,1], m);
		X.append(x);
	X = np.array(X);
	X = X.T;
	return X;

def computeY(X):
	m = -1;
	c = 0.5;
	y = m*X[:,1] + c
	y = np.where(X[:,2]<y,0,1)
	return y

def generateData(theta, xRange, m, noise):
	X  = generateX(xRange, m); #in excel sytle data (each row is a sample/example, contains multiple feature)
	y = computeY(X)
	#drawPlot(X, y);
	return X, y

def drawPlot(X, y):
	x1 = X[:,1]
	x2 = X[:,2]
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(x1,x2,y,c=y,marker='o')
	ax.set_xlabel('X1 Feature X')
	ax.set_ylabel('X2 Feature Y')
	ax.set_zlabel('Y  labels Z')
	#plt.plot(x1,x2,h)
	plt.show()

