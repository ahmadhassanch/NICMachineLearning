from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def genData(n,m):
	weights = [3, -5, -8, 12, 15];     # w1, w2, w3
	weights = weights[0:n]
	weights = np.asarray(weights)
	x = np.random.rand(n,m)		# matrix of inputs
	bias = -5;
	c = bias; 
	y = np.matmul(weights,x) + c;			# array of ground truths
	
	train_X = x[:,0:70]
	train_Y = y[0:70]
	test_X = x[:,70:]
	test_Y = y[70:]
	return train_X, train_Y, test_X, test_Y;
# Parameters
learning_rate = 0.1
training_epochs = 1000
m = 100;
n = 3;
train_X, train_Y, test_X, test_Y = genData(n,m);

# tf Graph Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.zeros([n,1]), name="weight")
b = tf.Variable(tf.cast(np.random.randn(1,1), tf.float32), name="bias")
Wx = tf.matmul(tf.transpose(W), X);
pred = tf.add(Wx, b)
cost = tf.reduce_mean(tf.square(pred-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    _, cost_, W_, b_, pred_ = sess.run([optimizer, cost, W, b, pred], feed_dict={X: train_X, Y: train_Y})
    if epoch % 100 == 0:
		print("Epoch:", epoch, "trainCost=", cost_, "W=", W_, "b=", b_)

#exit()
testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})  # same function as cost above
pred_TestY = sess.run(pred, feed_dict={X: test_X})   #Alternatively: #pred_TestY = W_ * test_X + b_
sess.close()

print("Epoch:", "trainCost=", cost_, "testCost=", testing_cost, "W=", W_, "b=", b_)


