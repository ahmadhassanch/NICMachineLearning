import numpy as np 
import tensorflow as tf


dataset_size = 3
weights = [3, 4];     # w1, w2, w3
input_length = len(weights);
bias = 2;

print "Hello"

def lossFunc(y,y_pred):
	squares = tf.square(y-y_pred)
	loss = tf.reduce_mean(squares)
	return loss


weights = np.asarray(weights)
x = np.random.rand(input_length,dataset_size)		# matrix of inputs
c = bias; #np.asarray([bias])

y = np.matmul(weights,x) + c;			# array of ground truths

print weights,x,c,y


# using x and y figure out a and b

with tf.Graph().as_default():
	x_holder = tf.placeholder(tf.float32, shape = (input_length, dataset_size))
	y_holder = tf.placeholder(tf.float32, shape = (dataset_size))

	suggested_m_shape = [1,input_length]
	suggested_m = tf.Variable(tf.truncated_normal(shape = suggested_m_shape, stddev = 1.0/input_length))
	suggested_c = tf.Variable(tf.truncated_normal(shape = [1,], stddev = 1.0/input_length))

	y_pred = tf.matmul(suggested_m,x_holder) + suggested_c

	loss = lossFunc(y_holder,y_pred)
	train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in xrange(100000):
			feeddict = {x_holder: x, y_holder:y}
			#print feeddict
			yret,lret,_,_suggested_m, _suggested_c =  sess.run([y_pred,loss,train_op,suggested_m, suggested_c], feeddict)
			if i % 100 == 0 :
				print i,lret,_suggested_m, _suggested_c
			if lret <.0000001 : 
				print "Converged", _suggested_m, _suggested_c
				break;

	print np.asarray(yret.shape)
	print np.asarray(lret)




