import numpy as np
import pdb
from sigmoid import sigmoid
#from hypothesis import hypothesis

def predictOneVsAll(all_theta, X):

	hyp = sigmoid( np.dot(X,all_theta.T))   # 5000,10 # gives the probability
	#hyp = hypothesis(all_theta, X);
											
	p = np.argmax(hyp, axis=1)	# argmax will apply max operation and returns the index
								# of the column(axis=1) in which max value lies but looking
								# though each example row-wise.
	#pdb.set_trace()
	return p

	# =========================================================================



