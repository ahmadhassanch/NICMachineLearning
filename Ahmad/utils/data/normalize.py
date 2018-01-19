import numpy as np

def normalize(data):
	dataMean = np.mean(data,axis=0)
	std = np.std(data,axis=0)
	x = np.subtract(data , dataMean)/std
	return x, dataMean, std;
