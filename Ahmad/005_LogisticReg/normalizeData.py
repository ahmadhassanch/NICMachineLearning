import numpy as np

def normalizeData(data):
    meanData = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    X = np.subtract(data,meanData)/std
    return X

def reverseNormalize(datax):
    z = x.dot(theta);
    h = sigmoid(z);
    return h
