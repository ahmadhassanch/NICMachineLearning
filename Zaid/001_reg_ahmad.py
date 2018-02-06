import numpy as np 
import tensorflow as tf

def lossFunc(y,y_pred):
	squares = tf.square(y-y_pred)
	loss = tf.reduce_mean(squares)
	return loss

dataset_size = 3
weights = [-3, 4];     
param_length = len(weights);
bias = 2;

weights = np.asarray(weights)
x = np.random.rand(param_length,dataset_size)		# matrix of inputs
c = bias; 

y = np.matmul(weights,x) + c;			# array of ground truths

hXin  = tf.placeholder('float')
hYout = tf.placeholder('float')

vW = tf.Variable(tf.zeros([1,param_length]));
vB = tf.Variable(tf.zeros([1]));
vYpred = tf.matmul(vW,hXin) + vB

loss = lossFunc(hYout,vYpred)
train_op = tf.train.AdamOptimizer(1).minimize(loss)
#train_op = tf.train.GradientDescentOptimizer(.5).minimize(loss)

init = tf.global_variables_initializer()
feeddict = {hXin: x, hYout:y}

sess = tf.Session()
sess.run(init)

for i in xrange(100000):
	yret,lret,_,_vW, _vB =  sess.run([vYpred,loss,train_op, vW, vB], feeddict)
	print i,lret,_vW, _vB
	if lret <.0000	001: break;

sess.close();

print "Converged Loss = ", lret, ", m = ", _vW,", c = ", _vB



