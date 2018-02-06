from sigmoid import sigmoid
import pdb


def hypothesis(theta, X):
	#pdb.set_trace()
	z=X.dot(theta)
	#return sigmoid(z)
	return sigmoid(z)