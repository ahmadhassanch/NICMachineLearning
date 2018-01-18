import numpy as np
from generateData import generateData
import lossAndHypothesis
import gradientDescent as gradDescent
import sys
sys.path.append('..')
import utils.CSV.CSVutils as csv


def defineRanges():
	# 2 feature example
	#theta  = np.array([.93, 0.025,0.5]);
	#noise  = np.array([0, 45,60]);
	#xRange = np.array([[1,1],[200, 2000],[20, 250]]);

	# 3 feature example
	#theta  = np.array([-2, 4, 2, 3]);
	#noise  = 1;
	#xRange = np.array([[1,1],[-1, 3],[-5, 2],[0, 1]]);

	theta = [];
	xRange = []
	theta.append(-2); xRange.append([1, 1]);
	theta.append(4); xRange.append([-1, 3]);
	theta.append(2); xRange.append([-5, 2]);
	theta.append(3); xRange.append([0, 1]);

	return theta, xRange, noise

def makeData(m):
	theta, xRange, noise= defineRanges()
	X, y, yIdeal = generateData(theta, xRange, noise,  m)
	#csv.writeCSV(X, y, '03_multiFeatureLinearReg.csv')
	#X, y = csv.readCSV('03_multiFeatureLinearReg.csv')
	return X, y, yIdeal


def main():
	m = 1000;

	theta, xRange, noise =  defineRanges()	
	N = theta.shape[0];                      # n = N -1  is the number of features
	X, y, yIdeal = makeData(m);
	
	# GRADIENT DESCENT
	nIterations = 500;
	alpha = 0.1
	thetaEst = np.zeros(N);   # t0, t1, .... , tn+1
	thetas, lossArr = gradDescent.gradientDescentLoop(X, y, thetaEst, alpha, nIterations)
	gradDescent.plotCost(thetas, lossArr)
	#thetaEst = thetas[-1];   #final value of thetas

main()

