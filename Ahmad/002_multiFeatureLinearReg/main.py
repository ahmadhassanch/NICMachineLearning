import numpy as np
import matplotlib.pyplot as plt
from generateData import generateData
import lossAndHypothesis
import gradientDescent as gradDescent

import sys
sys.path.append('..')

import utils.CSV.CSVutils as csv

def plotCost(cost):
	plt.plot(cost)
	plt.grid()
	plt.show()

def defineRanges():
	# 2 feature example
	#theta  = np.array([.93, 0.025,0.5]);
	#noise  = np.array([0, 45,60]);
	#xRange = np.array([[1,1],[200, 2000],[20, 250]]);

	# 3 feature example
	theta  = np.array([-2, 4, 2, 3]);
	noise  = 1;
	xRange = np.array([[1,1],[-1, 3],[-5, 2],[0, 1]]);
	return theta, xRange, noise

theta, xRange, noise = defineRanges()	
m = 100;
N = theta.shape[0];
n = N-1;


X, y, yIdeal = generateData(theta, xRange, noise, m)

#csv.writeCSV(X, y, '03_multiFeatureLinearReg.csv')
#X, y = csv.readCSV('03_multiFeatureLinearReg.csv')
print X.shape
print y.shape


thetaEst = np.zeros(N);   # t0, t1, .... , tn+1
alpha = 0.1
lossArr = [];
for i in range(50):
	thetaEst, loss = gradDescent.gradientDescent(i, X, y, alpha, thetaEst)
	lossArr.append(loss);


error = lossAndHypothesis.lossFunction(theta, X, y)
print "Error on Ideal Thetas: ", error
#iterationNo = np.arange()

plotCost(lossArr)


