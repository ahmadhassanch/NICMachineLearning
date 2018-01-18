import numpy as np
from generateData import generateData
import lossAndHypothesis
import gradientDescent as gradDescent
import sys
sys.path.append('..')
#import utils.CSV.CSVutils as csv


def defineRanges():
	theta = [];
	xRange = []
	theta.append(-2); xRange.append([1, 1]);
	theta.append(4); xRange.append([-1, 3]);
	theta.append(2); xRange.append([-5, 2]);
	theta.append(3); xRange.append([0, 1]);
	theta.append(-2.5); xRange.append([-2, 1]);
	return np.array(theta), np.array(xRange)

def makeData(m, computeYFunc, noise):
	thetaRef, xRange = defineRanges()
	X, y, yIdeal = generateData(thetaRef, xRange, noise,  m, computeYFunc)
	#csv.writeCSV(X, y, '03_multiFeatureLinearReg.csv')
	#X, y = csv.readCSV('03_multiFeatureLinearReg.csv')
	return X, y, yIdeal

def computeYNew(theta, X, noise):
	#y, yIdeal = computeY(theta,X,noise)
	m = X.shape[0];
	N = X.shape[1];

	yIdeal = lossAndHypothesis.hypothesis(theta, X);
	random = np.random.rand(m);
	yRand = 2 * noise * (random - 0.5);     #random = [0, 1]
	y = yIdeal + yRand; 

	return y, yIdeal


def main():
	m = 1000;
	#testCallback(newFunc)
	#exit()
	noise = 1;
	theta, xRange =  defineRanges()	
	N = theta.shape[0];                      # n = N -1  is the number of features
	X, y, yIdeal = makeData(m, computeYNew, noise);
	#X, y, yIdeal = makeDataMy(theta, xRange, @computeY, noise, m);
	
	# GRADIENT DESCENT
	nIterations = 500;
	alpha = 0.1
	thetaEst = np.zeros(N);   # t0, t1, .... , tn+1
	thetas, lossArr = gradDescent.gradientDescentLoop(X, y, thetaEst, alpha, nIterations)
	gradDescent.plotCost(thetas, lossArr)
	#thetaEst = thetas[-1];   #final value of thetas

main()

