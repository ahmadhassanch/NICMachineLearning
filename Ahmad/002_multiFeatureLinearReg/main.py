import numpy as np
import matplotlib.pyplot as plt
from generateData import generateData
import linearRegressionModel as linRegModel
import gradientDescent as gradDescent


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
xRange = np.array([[1,1],[-1, 3],[-5, 2]]);

m = 10;
N = theta.shape[0];
n = N-1;

X, y, yIdeal = generateData(theta, xRange, m, noise)
#X, y, yIdeal = generateDataLinReg();

thetaEst = np.zeros(N);   # t0, t1, .... , tn+1
alpha = 0.1
lossArr = [];
for i in range(50):
	thetaEst, loss = gradDescent.gradientDescent(i, X, y, alpha, thetaEst)
	lossArr.append(loss);


error = linRegModel.lossFunction(theta, X, y)
print "Error on Ideal Thetas: ", error
#iterationNo = np.arange()
plt.plot(lossArr)
plt.grid()
plt.show()



