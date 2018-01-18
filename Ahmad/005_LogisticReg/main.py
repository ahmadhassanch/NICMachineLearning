import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generateData import generateData2
import logisticRegression as logRegr
import gradientDescent as gradDescent

import sys
sys.path.append('..')

import utils.CSV.CSVutils as csv

#theta = [.93, 5.4, -3.42, 8.7];
#noise = 20;

#theta  = np.array([.93, 0.025,0.5]);
#noise  = np.array([0, 45,60]);
#xRange = np.array([[1,1],[200, 2000],[20, 250]]);

#theta  = np.array([5, 1] , -3]);
#noise  = np.array([0, 45, 55]);
#xRange = np.array([[1,1],[-1, 3], [2,5]]);
theta  = np.array([-2, 4, 2]);
noise  = 1;
xRange = np.array([[1,1],[-1, 1],[-1, 1]]);

m = 100;
N = theta.shape[0];
n = N-1;

X, y = generateData2(theta, xRange, m, noise)
#csv.writeCSV(X, y, '03_multiFeatureLinearReg.csv')


X, y = csv.readCSV('data/ex2data1.csv')
X = np.array([np.ones(len(y)), X[:,0] , X[:,1]]).T

x1 = X[:,1]
x2 = X[:,2]



#X, y, yIdeal = generateDataLinReg();

thetaEst = np.zeros(N);   # t0, t1, .... , tn+1
alpha = 0.00100
lossArr = [];
for i in range(90000):
	thetaEst, loss = gradDescent.gradientDescent(i, X, y, alpha, thetaEst)
	lossArr.append(loss);


error = logRegr.lossFunction(theta, X, y)
print "Error on Ideal Thetas: ", error

h = np.where(logRegr.hypothesis(thetaEst,X)>0.5,1,0)
h = logRegr.hypothesis(thetaEst,X)
dataX = np.array([h,y])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x1,x2,y,c=y,marker='o')
ax.set_xlabel('X1 Feature X')
ax.set_ylabel('X2 Feature Y')
ax.set_zlabel('Y  labels Z')
plt.plot(x1,x2,h)
plt.show()


#iterationNo = np.arange()
plt.plot(lossArr)
plt.grid()
plt.show()



