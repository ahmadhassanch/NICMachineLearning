import numpy as np
import matplotlib.pyplot as plt
import logisticRegression as logRegr

def generateDataPoly(xmin, xmax, m):
	x = np.linspace(xmin, xmax, m);
	np.random.shuffle(x)
	return x

def computeDataPoly(theta, x, noise):
	yIdeal = np.polyval(theta, x);
	random = np.random.rand(m);
	yrand = 2 * noise * (random - 0.5);     #random = [0, 1]
	y = yIdeal + yrand;
	return y, yIdeal;

def generateX(xRange, m):
	X = [];
	N = xRange.shape[0];
	for i in range(N):
		x = generateDataPoly(xRange[i,0], xRange[i,1], m);
		X.append(x);
	X = np.array(X);
	X = X.T;
	return X;


def computeY(theta, X, noise):
	#y, yIdeal = computeY(theta,X,noise)
	m = X.shape[0];
	N = X.shape[1];

	yIdeal = logRegr.hypothesis(theta, X);
	random = np.random.rand(m);
	yRand = 2 * noise * (random - 0.5);     #random = [0, 1]
	y = yIdeal + yRand;

	return y, yIdeal


def generateData(theta, xRange, m, noise):
	X  = generateX(xRange, m); #in excel sytle data (each row is a sample/example, contains multiple feature)
	y, yIdeal = computeY2(theta, X, noise)
	return X, y, yIdeal



def computeY2(X):
	y = -X[:,1] + 0.5
	y = np.where(X[:,2]<y,0,1)
	return y

def generateData2(theta, xRange, m, noise):
	X  = generateX(xRange, m); #in excel sytle data (each row is a sample/example, contains multiple feature)
	y = computeY2(X)
	return X, y


