import numpy as np 
import tensorflow as tf

input_length = 2		# this is a constant for this example
dataset_size = 100

def lossFunc(y,y_pred,w1):
	y_pred = tf.reshape(y_pred,[-1,])
	#if y_pred 
	# taking out the error cases
	epsilon = 0.000001
	y_pred = y_pred + tf.cast(y_pred < epsilon,tf.float32) * epsilon # I want these
	y_pred = y_pred - tf.cast(y_pred > (1-epsilon),tf.float32) * epsilon
	print y_pred.shape
	#y_pred = tf.multiply(y_pred,tf.cast(y_pred > 0.0001,tf.float32))
	bool_y = (tf.reduce_mean(tf.cast(y_pred < epsilon,tf.float32)), tf.reduce_mean(tf.cast(y_pred > (1-epsilon),tf.float32)))

	cost = -tf.reduce_mean(tf.multiply(y,tf.log(y_pred)) + tf.multiply((1-y),tf.log(1-y_pred)))# + tf.reduce_sum(tf.square(w1))
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = y_pred))
	return cost,bool_y

x = np.random.rand(input_length,dataset_size) * 10		# matrix of inputs
z = x[0] -3* x[1] - 10	# line defining decision boundary
f = 1/(1+np.exp(-z));
y = f > 0.5;


with tf.Graph().as_default():
	x_holder = tf.placeholder(tf.float32, shape = (input_length, dataset_size))
	y_holder = tf.placeholder(tf.float32, shape = (None))

	w1_shape = [1,input_length]
	w1 = tf.Variable(tf.truncated_normal(shape = w1_shape, stddev = 1.0/input_length))
	b1 = tf.Variable(tf.truncated_normal(shape = [1,], stddev = 1.0/input_length))

	y_pred = tf.nn.sigmoid(tf.matmul(w1,x_holder) + b1)

	loss,by = lossFunc(y_holder,y_pred,w1)
	train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in xrange(100000):
			feeddict = {x_holder: x, y_holder:y}
			#print feeddict
			yret,lret,_,wret,bret,booly =  sess.run([y_pred,loss,train_op,w1,b1,by], feeddict)
			if i % 100 == 0 :
				print i,lret,wret,bret,booly
			if (lret < .005): break

	print yret.shape
	print lret
	print np.mean(y == np.round(yret))

