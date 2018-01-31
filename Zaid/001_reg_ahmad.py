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
x_holder = tf.placeholder(tf.float32, x.shape)
y_holder = tf.placeholder(tf.float32, y.shape)

suggested_m_shape = [1,input_length]
print weights.shape
print suggested_m_shape
#exit()
tWeights = tf.zeros(suggested_m_shape);
suggested_c = tf.Variable(tf.zeros([1])

vWeights = tf.Variable(tWeights)

y_pred = tf.matmul(vWeights,x_holder) + suggested_c

loss = lossFunc(y_holder,y_pred)
train_op = tf.train.AdamOptimizer(10).minimize(loss)

init = tf.global_variables_initializer()
feeddict = {x_holder: x, y_holder:y}

with tf.Session() as sess:
	
	sess.run(init)

	for i in xrange(100000):
			yret,lret,_,_vWeights, _suggested_c =  sess.run([y_pred,loss,train_op,vWeights, suggested_c], feeddict)
			
			print i,lret,_vWeights, _suggested_c
			if lret <.0000001 : 
				print "Converged", _vWeights, _suggested_c
				break;

	print np.asarray(yret.shape)
	print np.asarray(lret)

print "Converged Loss = ", lret, ", m = ", _vWeights,", c = ", _suggested_c



