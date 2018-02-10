import numpy as np 
import tensorflow as tf

def lossFunc(y,y_pred):
	squares = tf.square(y-y_pred)
	loss = tf.reduce_mean(squares)
	return loss

m = 3
weights = [-3, 4];     
bias = 2;

n = len(weights);
weights = np.asarray(weights)
x = np.random.rand(n, m)		# matrix of inputs
yRef = np.matmul(weights,x) + bias;	# array of Yref

xin_holder  = tf.placeholder('float')
yout_holder = tf.placeholder('float')

W = tf.Variable(tf.zeros([1,n]));
B = tf.Variable(tf.zeros([1]));
yPred = tf.matmul(W,xin_holder) + B

loss = lossFunc(yout_holder, yPred)
#train_op = tf.train.AdamOptimizer(1).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(.5).minimize(loss)

init = tf.global_variables_initializer()
feeddict = {xin_holder: x, yout_holder:yRef}

sess = tf.Session()
sess.run(init)

for epoch in xrange(100000):
	yPred_,loss_,_,W_, B_ =  sess.run([yPred,loss,train_op, W, B], feeddict)
	print epoch,loss_,W_, B_
	if loss_ <.0000001: break;

sess.close();

print "Converged Loss = ", loss_, ", m = ", W_,", c = ", B_



