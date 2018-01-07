import numpy as np 
import tensorflow as tf

input_length = 2		# this is a constant for this example
dataset_size = 1000
num_classes = 2
hidden1_size = 6
hidden2_size = 6

print "Hello"

def lossFunc1(y,y_pred,w1):
	y_pred = tf.reshape(y_pred,[dataset_size,])

	#if y_pred 
	# taking out the error cases
	epsilon = 0.000001
	y_pred = y_pred + tf.cast(y_pred < epsilon,tf.float32) * epsilon # I want these
	y_pred = y_pred - tf.cast(y_pred > (1-epsilon),tf.float32) * epsilon
	print y_pred.shape
	#y_pred = tf.multiply(y_pred,tf.cast(y_pred > 0.0001,tf.float32))
	bool_y = (tf.reduce_mean(tf.cast(y_pred < epsilon,tf.float32)), tf.reduce_mean(tf.cast(y_pred > (1-epsilon),tf.float32)))

	cost = tf.reduce_mean(-(tf.multiply(y,tf.log(y_pred)) + tf.multiply((1-y),tf.log(1-y_pred))))# + tf.reduce_sum(tf.square(w1))
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = y_pred))
	return cost,bool_y

def lossFunc(y,logits):

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = y))
	return loss





x = np.random.rand(input_length,dataset_size) 		# matrix of inputs
y = x[1] - np.sqrt(1 - np.square(x[0]))				# line defining decision boundary
#y = x[1] + x[0] - 10
y = y > 0

print y.shape




# using x and y figure out a and b
wreturned1 = []
wreturned2 = []

with tf.Graph().as_default():
	x_holder = tf.placeholder(tf.float32, shape = (input_length,dataset_size))
	y_holder = tf.placeholder(tf.int32, shape = (dataset_size))
	

	w1_shape = [input_length, hidden1_size]
	w1 = tf.Variable(tf.truncated_normal(shape = w1_shape, stddev = 2.0/(input_length + hidden1_size)))
	b1 = tf.Variable(tf.truncated_normal(shape = [1,hidden1_size], stddev = 2.0/(input_length + hidden1_size)))
	
	h1 = tf.nn.relu(tf.matmul(tf.transpose(x_holder),w1) + b1)

	w2_shape = [hidden1_size, hidden2_size]
	w2 = tf.Variable(tf.truncated_normal(shape = w2_shape, stddev = 2.0/(hidden1_size+ hidden2_size)))
	b2 = tf.Variable(tf.truncated_normal(shape = [1,hidden2_size], stddev = 2.0/(hidden1_size+ hidden2_size)))

	h2 = tf.nn.relu(tf.matmul(h1,w2)) + b2

	w3_shape = [hidden2_size, num_classes]
	w3 = tf.Variable(tf.truncated_normal(shape = w3_shape, stddev = 2.0/(hidden2_size+num_classes)))
	b3 = tf.Variable(tf.truncated_normal(shape = [1,num_classes], stddev = 2.0/(hidden2_size+num_classes)))
	
	logits = tf.matmul(h2,w3) + b3

	
	loss = lossFunc(y_holder,logits)
	train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
	#train_op = tf.train.AdagradOptimizer(0.005).minimize(loss)


	#correct_prediction = tf.equal(tf.cast(y,tf.int32), tf.cast(tf.argmax(logits,0),tf.int32))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in xrange(50000):
			feeddict = {x_holder: x, y_holder:y}
			#print feeddict
			logits_ret,lret,_,wret,bret, =  sess.run([logits,loss,train_op,w1,b1,], feeddict)
			if i % 1000 == 0 :
				print i,lret
				
				

	print logits_ret.shape
	yret =  np.asarray((np.argmax(logits_ret,axis=1)))
	print np.asarray(lret)
	print yret.shape
	print y.shape
	print np.mean(y == yret)

xlist = [column(wreturned1,0),column(wreturned2,0)]
ylist = [column(wreturned1,1),column(wreturned2,1)]

animateMultiple.animatePlots(xlist, ylist)
