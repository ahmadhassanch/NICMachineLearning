import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from plots import animateMultiple

input_length = 2		# this is a constant for this example
dataset_size = 1000
num_classes = 2

print "Hello"

def column(matrix, i):
    return [row[i] for row in matrix]

def lossFunc1(y,y_pred,w1):
	y_pred = tf.reshape(y_pred,[dataset_size,])

	#if y_pred 
	# taking out the error cases
	epsilon = 1e-10
	y_pred = y_pred + tf.cast(y_pred < epsilon,tf.float32) * epsilon # I want these
	y_pred = y_pred - tf.cast(y_pred > (1-epsilon),tf.float32) * epsilon
	print y_pred.shape
	#y_pred = tf.multiply(y_pred,tf.cast(y_pred > 0.0001,tf.float32))
	bool_y = (tf.reduce_mean(tf.cast(y_pred < epsilon,tf.float32)), tf.reduce_mean(tf.cast(y_pred > (1-epsilon),tf.float32)))

	cost = tf.reduce_mean(-(tf.multiply(y,tf.log(y_pred)) + tf.multiply((1-y),tf.log(1-y_pred)))) + tf.reduce_sum(tf.square(w1))
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = y_pred))
	return cost,bool_y

def lossFunc(y,logits):

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = y))
	return loss




x = np.random.rand(input_length,dataset_size) * 10		# matrix of inputs
y = x[1] + x[0] - 10	# line defining decision boundary
y = y > 0




print  y.astype(int) 



# using x and y figure out a and b
wreturned1 = []
wreturned2 = []

with tf.Graph().as_default():
	x_holder = tf.placeholder(tf.float32, shape = (input_length,dataset_size))
	y_holder = tf.placeholder(tf.int32, shape = (dataset_size))
	

	w1_shape = [input_length, num_classes]
	w1 = tf.Variable(tf.truncated_normal(shape = w1_shape, stddev = 1.0/input_length))
	b1 = tf.Variable(tf.truncated_normal(shape = [1,num_classes], stddev = 1.0/input_length))

	logits = tf.matmul(tf.transpose(x_holder),w1) + b1

	loss = lossFunc(y_holder,logits)
	train_op = tf.train.AdamOptimizer(3e-4).minimize(loss)

	#correct_prediction = tf.equal(tf.cast(y,tf.int32), tf.cast(tf.argmax(logits,0),tf.int32))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in xrange(10000):
			feeddict = {x_holder: x, y_holder:y}
			#print feeddict
			logits_ret,lret,_,wret,bret, =  sess.run([logits,loss,train_op,w1,b1,], feeddict)
			if i % 100 == 0 :
				print i,lret,wret,bret
				wreturned1.append(wret[:,0])
				wreturned2.append(wret[:,1])

	print logits_ret.shape
	yret =  np.asarray((np.argmax(logits_ret,axis=1)))
	print np.asarray(lret)
	print yret.shape
	print y.shape
	print np.mean(y == yret)

# removing regularization will give zero loss

f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(column(wreturned1,0), column(wreturned1,1),"*-")
ax2.plot(column(wreturned2,0), column(wreturned2,1),"*-")

plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('Weights convergence with regularization(Softmax)')
plt.grid(True)

plt.savefig("testSoft2.png")
plt.show()

xlist = [column(wreturned1,0),column(wreturned2,0)]
ylist = [column(wreturned1,1),column(wreturned2,1)]

animateMultiple.animatePlots(xlist, ylist)
