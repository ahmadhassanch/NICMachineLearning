import numpy as np
import matplotlib.pyplot as plt
from generateData import generateData
import lossAndHypothesis as lossAndHypothesis
import gradientDescent as gradDescent
import normalizeData
import time
import sys
sys.path.append('..')

import utils.CSV.CSVutils as csv


def drawPlot(X, y):
	x1 = X[:,1]
	x2 = X[:,2]
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(x1,x2,y,c=y,marker='o')
	ax.set_xlabel('X1 Feature X')
	ax.set_ylabel('X2 Feature Y')
	ax.set_zlabel('Y  labels Z')
	#plt.plot(x1,x2)
	plt.show()


start_time = time.time()

theta  = np.array([-2, 4, 2]);
noise  = 1;
xRange = np.array([[1,1],[-1, 1],[-1, 1]]);

m = 100;
N = theta.shape[0];
n = N-1;

X, y = generateData(theta, xRange, m, noise)
x1 = X[:,1]
x2 = X[:,2]
x = np.linspace(-1, 1, 100);
f = - x + 0.5;

plt.plot(x1[y==1], x2[y==1], 'g*')
plt.plot(x1[y==0], x2[y==0], 'r*')
#plt.plot(x,f)
#plt.show()
#exit()
#csv.writeCSV(X, y, '03_multiFeatureLinearReg.csv')


#X, y = csv.readCSV('data/ex2data1.csv')

#X = normalizeData.normalizeData(X)
#X = np.array([np.ones(len(y)), X[:,0] , X[:,1]]).T

x1 = X[:,1]
x2 = X[:,2]
#plt.plot(x1, x2)




#X, y, yIdeal = generateDataLinReg();

thetaEst = np.zeros(N);   # t0, t1, .... , tn+1
alpha = 0.9
lossArr = [];
for i in range(5000):
	thetaEst, loss = gradDescent.gradientDescent(i, X, y, alpha, thetaEst)
	lossArr.append(loss);


error = lossAndHypothesis.lossFunction(theta, X, y)
print "Error on Ideal Thetas: ", error

h = np.where(lossAndHypothesis.hypothesis(thetaEst,X)>0.5,1,0)
h = lossAndHypothesis.hypothesis(thetaEst,X)
dataX = np.array([h,y])

#plt.scatter(x1,x2)
#plt.plot(x1, x2)

#plt.show()
print time.time() - start_time , " s"
t0 = thetaEst[0]
t1 = thetaEst[1]
t2 = thetaEst[2]
#iterationNo = np.arange()
#plt.plot(lossArr)
#plt.grid()
#plt.show()

m = -t1/t2;
c = -t0/t2;
print m, c
f = m* x + c;
plt.plot(x,f)

#plt.plot(x,f)
plt.show()


