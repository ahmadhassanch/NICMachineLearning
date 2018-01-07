import numpy as np 
import tensorflow as tf


input_length = 2		# this is a constant for this example
dataset_size = 1000
batch_size = 100
num_classes = 1
hidden1_size = 6
hidden2_size = 6

print "Hello"
scale = 2
bias = 0.5


x = np.random.rand(dataset_size,input_length) - bias # matrix of inputs
xstore = x.reshape(1,dataset_size * input_length)
np.savetxt("x.csv", xstore, delimiter=",")

w1 = (np.random.rand(input_length,hidden1_size) - bias) 
w1 = w1 * scale 
b1 = np.random.rand(1,hidden1_size) - bias
b1 = b1 * scale

h1 = np.matmul(x,w1) + b1
h1.clip(min = 0.0)

w2 = np.random.rand(hidden1_size,hidden2_size) - bias 
w2 = w2 * scale
b2 = np.random.rand(1,hidden2_size) - bias
b2 = b2 * scale

h2 = np.matmul(h1,w2) + b2
h1.clip(min = 0.0)

w3 = np.random.rand(hidden2_size,num_classes) - bias 
w3 = w3 * scale
b3 = np.random.rand(1,num_classes) - bias
b3 = b3 * scale

logits = np.matmul(h2,w3) + b3

y = logits > 0.
#y = np.argmax(logits,axis = 1)
ystore = y.reshape(1,dataset_size)
np.savetxt("y.csv", ystore, delimiter=",")
print y.shape

print np.mean(y==True)

def lossFunc(y,logits):

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = y))
	return loss

def lossFunc1(y,y_pred):
	y_pred = tf.reshape(y_pred,[-1])

	#if y_pred 
	# taking out the error cases
	epsilon = 1e-10
	y_pred = y_pred + tf.cast(y_pred < epsilon,tf.float32) * epsilon # I want these
	y_pred = y_pred - tf.cast(y_pred > (1-epsilon),tf.float32) * epsilon
	print y_pred.shape
	#y_pred = tf.multiply(y_pred,tf.cast(y_pred > 0.0001,tf.float32))
	bool_y = (tf.reduce_mean(tf.cast(y_pred < epsilon,tf.float32)), tf.reduce_mean(tf.cast(y_pred > (1-epsilon),tf.float32)))

	cost = tf.reduce_mean(-(tf.multiply(y,tf.log(y_pred)) + tf.multiply((1-y),tf.log(1-y_pred))))# + tf.reduce_sum(tf.square(w1))
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = y_pred))
	return cost #,bool_y


with tf.Graph().as_default():
	x_holder = tf.placeholder(tf.float32, shape = (None,input_length))
	y_holder = tf.placeholder(tf.float32, shape = (None))
	

	w1_shape = [input_length, hidden1_size]
	w1 = tf.Variable(tf.truncated_normal(shape = w1_shape, stddev = 2.0/(input_length + hidden1_size)))
	b1 = tf.Variable(tf.truncated_normal(shape = [1,hidden1_size], stddev = 2.0/(input_length + hidden1_size)))
	
	h1 = tf.nn.relu(tf.matmul(x_holder,w1) + b1)

	w2_shape = [hidden1_size, hidden2_size]
	w2 = tf.Variable(tf.truncated_normal(shape = w2_shape, stddev = 2.0/(hidden1_size+ hidden2_size)))
	b2 = tf.Variable(tf.truncated_normal(shape = [1,hidden2_size], stddev = 2.0/(hidden1_size+ hidden2_size)))

	h2 = tf.nn.relu(tf.matmul(h1,w2)) + b2

	w3_shape = [hidden2_size, num_classes]
	w3 = tf.Variable(	tf.truncated_normal(shape = w3_shape, stddev = 2.0/(hidden2_size+num_classes)))
	b3 = tf.Variable(tf.truncated_normal(shape = [1,num_classes], stddev = 2.0/(hidden2_size+num_classes)))
	
	logits = tf.sigmoid(tf.matmul(h2,w3) + b3)

	
	loss = lossFunc1(y_holder,logits)
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
	#train_op = tf.train.AdagradOptimizer(0.005).minimize(loss)
	ypred = tf.squeeze(logits > 0.5)
	correct_prediction = tf.equal(tf.cast(y_holder,tf.bool), ypred)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#correct_prediction = tf.equal(tf.cast(y,tf.int32), tf.cast(tf.argmax(logits,0),tf.int32))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	batches = dataset_size // batch_size
	for i in xrange(5000):
		for j in xrange(batches):
			indexes = range(j*batch_size,(j+1)*batch_size)
			feeddict = {x_holder: x[indexes,:], y_holder:y[indexes]}
			logits_ret,lret,_,wret,bret,acc,y_ =  sess.run([logits,loss,train_op,w1,b1,accuracy,ypred], feeddict)

		if i % 100 == 0 :
			feeddict = {x_holder: x, y_holder:y}			
			logits_ret_final,lret,_,wret,bret,y_,acc =  sess.run([logits,loss,train_op,w1,b1,ypred,accuracy], feeddict)		
			print i,lret,acc
			print 'outer acc', np.mean(y == y_)
			print y_.shape, logits_ret.shape

	feeddict = {x_holder: x, y_holder:y}			
	logits_ret_final,lret,_,wret,bret,y_,acc =  sess.run([logits,loss,train_op,w1,b1,ypred,accuracy], feeddict)			
	print 'inner acc', acc
	print 'outer acc', np.mean(y == y_)



