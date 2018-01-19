import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
import utils.gradient.gradientDescent as gradDescent
import utils.data.generateData as genData
import utils.CSV.CSVutils as csv
import utils.data.normalize as norm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def sigmoid(z):
	gx = 1/(1+np.exp(-z))
	return gx

def hypFunc(theta, x):
	ht = x.dot(theta);
	h = sigmoid(ht)
	return h;


def lossFunc(theta, X, y):
	global i;
	m = X.shape[0];
	h  = hypFunc(theta, X);
	p1 = -y*np.log(h)
	p2 = (1-y)* np.log(1.0-h)
	J = np.mean(p1 - p2)
	print i, theta, " == ", J
	i = i + 1;
	return J;


def defineThetaAndRanges():
	theta = [];
	xRange = []
	theta.append(-2); xRange.append([1, 1]);
	theta.append(3); xRange.append([-1, 1]);
	theta.append(2); xRange.append([-1, 1]);
	theta.append(5); xRange.append([-1, 1]);
	return np.array(theta), np.array(xRange)


def computeY(theta, X, noise):
	#xt = X.dot(theta);
	yt = hypFunc(theta, X)
	#-X[:,1] + 0.5
	y = np.where(yt<0.5,0,1)
	yIdeal = np.ones(y.shape)
	return y, yIdeal


i = 0;

def plot3d(x1, x2, x3, y):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(x1,x2,x3,c=y,marker='o')
	ax.set_xlabel('X1 Feature X')
	ax.set_ylabel('X2 Feature Y')
	ax.set_zlabel('X3 Feature Z')
	#plt.plot(x1,x2,h)

#ax = Axes3D(fig)
	x = [-1,1,1,-1]
	y = [-1,-1,1,1]
	z =[.5, .5, .5, .5]
	verts = [zip(x, y,z)]
	ax.add_collection3d(Poly3DCollection(verts))

	plt.show()

def main():
	m = 100;
	noise = 0.0;
	thetaRef, xRange = defineThetaAndRanges()
	X, y, yIdeal = genData.generateData(thetaRef, xRange, noise, m, computeY)

	plot3d(X[:,1],X[:,2], X[:,3],y)
	exit()
	#X = norm.normalize(X)
	N = X[0].shape[0]  # n = N -1  is the number of features

	nIter = 7000;
	alpha = 0.9
	thetaEst = np.zeros(N);  # t0, t1, .... , tn+1
	thetas, lossArr = gradDescent.gradientDescentLoop(hypFunc, lossFunc, X, y, thetaEst, alpha, nIter)

	#gradDescent.plotCost(thetas, lossArr, True)
	#gradDescent.plotCost3D(thetas, lossArr,X,y, True)
	thetaFinal = thetas[-1];
	print i, '\n ThetaRef', thetaRef, '\n ThetaEst', thetas[-1], '\n Error', thetaFinal - thetaRef

	x = xRange[1] ;
	
	
	plt.scatter(X[:,1],X[:,2],c=y)
	m = -thetaFinal[1]/thetaFinal[2]
	c = -thetaFinal[0]/thetaFinal[2]
	print "m", m
	print "c", c
	f = (m*x+c);

	plt.plot(x,f,'b+-')
	plt.show()

main()

