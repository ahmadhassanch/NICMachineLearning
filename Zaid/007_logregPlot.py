import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

input_length = 2		# this is a constant for this example
dataset_size = 1000

print "Hello"

def lossFunc(y,y_pred,w1):
	y_pred = tf.reshape(y_pred,[dataset_size,])
	#if y_pred 
	# taking out the error cases
	epsilon = 0.000001
	y_pred = y_pred + tf.cast(y_pred < epsilon,tf.float32) * epsilon # I want these
	y_pred = y_pred - tf.cast(y_pred > (1-epsilon),tf.float32) * epsilon
	print y_pred.shape
	#y_pred = tf.multiply(y_pred,tf.cast(y_pred > 0.0001,tf.float32))
	bool_y = (tf.reduce_mean(tf.cast(y_pred < epsilon,tf.float32)), tf.reduce_mean(tf.cast(y_pred > (1-epsilon),tf.float32)))

	cost = tf.reduce_mean(-(tf.multiply(y,tf.log(y_pred)) + tf.multiply((1-y),tf.log(1-y_pred)))) + tf.reduce_sum(tf.square(w1))
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = y_pred))
	return cost,bool_y





x = np.random.rand(input_length,dataset_size) * 10		# matrix of inputs
y = x[1] + x[0] - 10	# line defining decision boundary
y = y > 0



print  y.astype(int) 

wreturned = []

# using x and y figure out a and b

with tf.Graph().as_default():
	x_holder = tf.placeholder(tf.float32, shape = (input_length, dataset_size))
	y_holder = tf.placeholder(tf.float32, shape = (dataset_size))

	w1_shape = [1,input_length]
	w1 = tf.Variable(tf.truncated_normal(shape = w1_shape, stddev = 1.0/input_length))
	b1 = tf.Variable(tf.truncated_normal(shape = [1,], stddev = 1.0/input_length))

	y_pred = tf.nn.sigmoid(tf.matmul(w1,x_holder) + b1)

	loss,by = lossFunc(y_holder,y_pred,w1)
	train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in xrange(20000):
			feeddict = {x_holder: x, y_holder:y}
			#print feeddict
			yret,lret,_,wret,bret,booly =  sess.run([y_pred,loss,train_op,w1,b1,by], feeddict)
			if i % 100 == 0 :
				print i,lret,wret,bret,booly
				wreturned.append(wret[0])

	print np.asarray(yret.shape)
	print np.asarray(lret)
	print np.mean(y == np.round(yret))
	print wreturned[0]

# removing regularization will give zero loss
def column(matrix, i):
    return [row[i] for row in matrix]

plt.plot(column(wreturned,0), column(wreturned,1),"*-")

plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('Weights convergence with regularization')
plt.grid(True)
plt.savefig("test.png")
plt.show()
