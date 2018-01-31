import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
import utils.gradient.gradientDescent as gradDescent
import utils.data.generateData as genData
import utils.data.normalize as norm

def hypFunc(theta, x):
	z = x.dot(theta);
	gx = 1/(1+np.exp(-z))
	return gx

def lossFunc(theta, X, y):
	global i;
	m = X.shape[0];
	h  = hypFunc(theta, X);
	p1 = -y*np.log(h)
	p2 = (1-y)* np.log(1 - h)
	J = np.mean(p1 - p2)
	thetaNew = theta;
	#thetaNew = thetaNew / thetaNew[0];
	print i, thetaNew, " == ", J
	i = i + 1;
	return J;

def defineThetaAndRanges():
	theta = [];
	xRange = []
	theta.append(-2.0); xRange.append([1, 1]);
	theta.append(3.0); xRange.append([-1, 1]);
	theta.append(-2.0); xRange.append([-1, 1]);
	theta.append(5.0); xRange.append([-1, 1]);

	theta = theta / np.max(np.abs(theta));
	return np.array(theta), np.array(xRange)

def computeY(theta, X, noise):
	yt = hypFunc(theta, X)
	#print X.shape
	y = np.where(yt<0.5,0,1)

	step = int(X.shape[0]* noise) 
	
	for i in range(step):
		r = int(np.random.rand()*X.shape[0])
		y[r] = 1 - y[r];

	yIdeal = np.ones(y.shape)
	return y, yIdeal



def main():
	m = 300;
	noise = 0.0;  # enter percent noise
	thetaRef, xRange = defineThetaAndRanges()
	X, y, yIdeal = genData.generateData(thetaRef, xRange, noise, m, computeY)

	X = norm.normalize(X)

	N = X[0].shape[0]  # n = N -1  is the number of features

	nIter = 1000;
	alpha = .9
	thetaEst = np.random.rand(N)*.00001;  # t0, t1, .... , tn+1
	thetas, lossArr = gradDescent.gradientDescentLoop(hypFunc, lossFunc, X, y, thetaEst, alpha, nIter)

	gradDescent.plotCost(thetas, lossArr, False)

	thetaFinal = thetas[-1];
	print i, '\n ThetaRef', thetaRef, '\n ThetaEst', thetas[-1], '\n Error', thetaFinal - thetaRef

	x = xRange[1] ;
	print thetaRef / np.max(np.abs(thetaRef))
	print thetaFinal / np.max(np.abs(thetaFinal))
	n = N -1;
	if(n==3):  #3 feature case.
		gradDescent.plotDataAndDecisionBoundary3D(X, thetaFinal, y, True)

i = 0
main()

