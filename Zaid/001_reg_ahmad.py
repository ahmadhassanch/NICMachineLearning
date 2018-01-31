import numpy as np 
import tensorflow as tf

def lossFunc(y,y_pred):
	squares = tf.square(y-y_pred)
	loss = tf.reduce_mean(squares)
	return loss

dataset_size = 3
weights = [-3, 4];     
input_length = len(weights);
bias = 2;

weights = np.asarray(weights)
x = np.random.rand(input_length,dataset_size)		# matrix of inputs
c = bias; #np.asarray([bias])

y = np.matmul(weights,x) + c;			# array of ground truths

# using x and y figure out a and b
hX = tf.placeholder(tf.float32, x.shape)
hY = tf.placeholder(tf.float32, y.shape)

suggested_m_shape = [1,input_length]

tWeights = tf.zeros(suggested_m_shape);
tBias = tf.zeros([1]);

vB = tf.Variable(tBias)
vW = tf.Variable(tWeights)

vYref = tf.matmul(vW,hX) + vB

loss = lossFunc(hY,vYref)
train_op = tf.train.AdamOptimizer(.1).minimize(loss)

init = tf.global_variables_initializer()
feeddict = {hX: x, hY:y}

with tf.Session() as sess:
	
	sess.run(init)

	for i in xrange(100000):
			yret,lret,_,_vW, _vB =  sess.run([vYref,loss,train_op, vW, vB], feeddict)
			
			print i,lret,_vW, _vB
			if lret <.00001 : 
				print "Converged", _vW, _vB
				break;

	print np.asarray(yret.shape)
	print np.asarray(lret)

print "Converged Loss = ", lret, ", m = ", _vW,", c = ", _vB



