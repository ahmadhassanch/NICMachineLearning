import scipy.io
import numpy as np
import random
import pdb

from displayData import displayData
from trainOneVsAll import trainOneVsAll
from predictOneVsAll import predictOneVsAll

numClasses = 10;          # 10 labels, from 1 to 10

data = scipy.io.loadmat('ex3data1.mat'); # training data stored in arrays X, y
X = np.array(data['X'])                  # 5000 x 400   (20x20 size of one image)
y = np.array(data['y']%10)                  # 5000 x 1      labels
m =  len(y)

rand_indices = random.sample(range(0,m), 100)
sel = X[rand_indices, :]
displayData(sel);

Lambda = 0.1;
print X.shape
X = np.column_stack((np.ones((m,1)), X))
print X.shape

y=y.flatten()                            # 5000 x 1 to 5000
all_theta = trainOneVsAll(X, y, numClasses, Lambda); 

pred = predictOneVsAll(all_theta, X);

#pdb.set_trace()
mean = np.mean((pred==y).astype(int))
print pred
print y
err = (pred == y).astype(int)
k = 2200;
print err[k:k+100]
print pred[k:k+100]
print y[k:k+100]
acc = mean*100
print 'Accuracy: ', acc

