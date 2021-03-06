import numpy as np
import sys
sys.path.append('..')
import utils.gradient.gradientDescent as gradDescent
import utils.data.generateData as genData
import utils.CSV.CSVutils as csv
import utils.data.normalize as norm


def hypFunc(theta, x):
	h = x.dot(theta);
	return h;

def lossFunc(theta, X, y):
	global i
	i = i+1;
	m = X.shape[0];
	h  = hypFunc(theta, X);
	loss = 0.5/m*np.sum((h - y)**2)
	#print i, theta, " == ", loss
	return loss;

def defineThetaAndRanges():
	theta = [];
	xRange = []
	theta.append(-2); xRange.append([1, 1]);
	theta.append(4); xRange.append([-1, 3]);
	theta.append(2); xRange.append([-5, 2]);
	return np.array(theta), np.array(xRange)

def computeY(theta, X, noise):
	m = X.shape[0];
	N = X.shape[1];

	yIdeal = hypFunc(theta, X);
	#random = np.random.rand(m);
	random = np.random.normal(0,1,yIdeal.shape[0]);
	yRand = 2 * noise * (random - 0.5);     #random = [0, 1]
	y = yIdeal + yRand; 
	return y, yIdeal


i = 0;
def main():
	m = 1000;
	noise = 1;
	thetaRef, xRange = defineThetaAndRanges()
	X, y, yIdeal = genData.generateData(thetaRef, xRange, noise,  m, computeY)
	X = norm.normalize(X)

	N = X[0].shape[0]                    # n = N -1  is the number of features

	nIter = 100;
	alpha = 0.1
	thetaEst = np.zeros(N);   # t0, t1, .... , tn+1
	thetas, lossArr = gradDescent.gradientDescentLoop(hypFunc, lossFunc, X, y, thetaEst, alpha, nIter)
	thetaFinal = thetas[-1];
	gradDescent.plotCost(thetas, lossArr, False)
	print i, '\n ThetaRef', thetaRef,'\n ThetaEst', thetaFinal,'\n Error', thetaFinal - thetaRef
	
	n = N -1;
	if(n==2):  #3 feature case.
		gradDescent.plotDataAndRegressionPlane3D(X, thetaFinal, y, True)

	print n
main()

