import numpy as np

def normalize(X):
	#print X
	maxX = np.max(X,axis=0)
	minX = np.min(X,axis=0)
	#print minX
	midX = (maxX + minX)/2
	#print midX
	rangeX = maxX - minX
	#print "rangeX", rangeX
	#print X-midX

	for i in range(1,X.shape[1]):
		#print "ri", rangeX[i]
		#print X[:,i]
		X[:,i] = 2*(X[:,i]-midX[i])/rangeX[i]

	#print X
	#print y
	return X;
