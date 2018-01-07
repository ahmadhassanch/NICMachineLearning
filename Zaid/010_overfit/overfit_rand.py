import numpy as np 
import time

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randint
import os

input_length = 2		# this is a constant for this example
dataset_size = 150
testset_size = 100
batch_size = 5
num_classes = 1
hidden1_size = 400
hidden2_size = 400

print "Hello"

def lossFunc(y,y_pred):
	#if y_pred 
	# taking out the error cases
	epsilon = 1e-6
	y_pred = y_pred + tf.cast(y_pred < epsilon,tf.float32) * epsilon # I want these
	y_pred = y_pred - tf.cast(y_pred > (1-epsilon),tf.float32) * epsilon
	print y_pred.shape
	#y_pred = tf.multiply(y_pred,tf.cast(y_pred > 0.0001,tf.float32))
	bool_y = (tf.reduce_mean(tf.cast(y_pred < epsilon,tf.float32)), tf.reduce_mean(tf.cast(y_pred > (1-epsilon),tf.float32)))

	cost = tf.reduce_mean(-(tf.multiply(y,tf.log(y_pred)) + tf.multiply((1-y),tf.log(1-y_pred))))# + tf.reduce_sum(tf.square(w1))
	tf.summary.scalar('cost', cost)
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = y_pred))
	return cost #,bool_y






x = np.fromfile('input.csv',sep=',')	# matrix of inputs
x = x.reshape(dataset_size, input_length)
y = np.fromfile('y.csv',sep=',')
y = y.reshape(dataset_size,1)




xval = np.fromfile('xval.csv',sep=',')	# matrix of inputs
print xval.shape
xval = xval.reshape(testset_size, input_length)
print xval.shape
yval = np.fromfile('yval.csv',sep=',')
print yval.shape
yval = yval.reshape(testset_size,1)
print yval.shape




#xtest = np.random.rand(testset_size,input_length) * 10		# matrix of inputs
nx, ny = (57, 51)
x_coords = np.linspace(0, 10, nx)
y_coords = np.linspace(0, 10, ny)
xtest1, xtest2 = np.meshgrid(x_coords, y_coords)
xtest1 = xtest1.reshape(xtest1.size)
xtest2 = xtest2.reshape(xtest2.size)
xtest = np.asarray([xtest1, xtest2])
xtest = xtest.transpose()
ytest = xtest[:,1] + xtest[:,0] - 10	# line defining decision boundary
ytest = ytest > 0
ytest = ytest.reshape(len(ytest),1)


# using x and y figure out a and b
# using x and y figure out a and b
wreturned1 = []
wreturned2 = []

g = tf.Graph()

with g.as_default():
	x_holder = tf.placeholder(tf.float32, shape = (None,input_length))
	y_holder = tf.placeholder(tf.float32, shape = (None,1))
	train_holder = tf.placeholder(tf.bool) 
	

	w1_shape = [input_length, hidden1_size]
	w1 = tf.Variable(tf.truncated_normal(shape = w1_shape, stddev = 2.0/(input_length + hidden1_size)))
	b1 = tf.Variable(tf.truncated_normal(shape = [1,hidden1_size], stddev = 2.0/(input_length + hidden1_size)))
	
	h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_holder,w1) + b1),0.5)
	#h1 = tf.layers.dropout(h1,rate = 0.5, training = train_holder)

	w2_shape = [hidden1_size, hidden2_size]
	w2 = tf.Variable(tf.truncated_normal(shape = w2_shape, stddev = 2.0/(hidden1_size+ hidden2_size)))
	b2 = tf.Variable(tf.truncated_normal(shape = [1,hidden2_size], stddev = 2.0/(hidden1_size+ hidden2_size)))

	h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1,w2) + b2),0.5)
	#h2 = tf.layers.dropout(h2,rate = 0.5, training = train_holder)

	w3_shape = [hidden2_size, num_classes]
	w3 = tf.Variable(tf.truncated_normal(shape = w3_shape, stddev = 2.0/(hidden2_size+num_classes)))
	b3 = tf.Variable(tf.truncated_normal(shape = [1,num_classes], stddev = 2.0/(hidden2_size+num_classes)))
	
	logits = tf.matmul(h2,w3) + b3

	probs = tf.sigmoid(logits)

	
	loss = lossFunc(y_holder,probs)
	tf.summary.scalar('loss', loss)
	train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
	#train_op = tf.train.AdagradOptimizer(0.005).minimize(loss)
	ypred = probs > 0.5
	correct_prediction = tf.equal(tf.cast(y_holder,tf.bool), ypred)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	summary = tf.summary.merge_all()
	init = tf.global_variables_initializer()
	sess = tf.Session()
	summary_writer = tf.summary.FileWriter(os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S")), sess.graph)
	sess.run(init)

	batches = dataset_size // batch_size
	for i in xrange(20000):
		for j in xrange(batches):
			feeddict = {x_holder: x, y_holder:y, train_holder: True}
			probs_ret,lret,_ ,acc=  sess.run([probs,loss,train_op,accuracy], feeddict)
		if (i+1) % 10 == 0 :
			print (i+1),lret,acc
			summary_str = sess.run(summary, feed_dict=feeddict)
			summary_writer.add_summary(summary_str,i)
			summary_writer.flush()


	yret = (probs_ret > 0.5)
	print 'train accuracy: ' , np.mean(y == yret) , acc
	
	#feeddict_new = {x_holder: xval, y_holder:yval, train_holder: False}
	#probs_val,pred,acc,pred_y =  sess.run([probs,correct_prediction,accuracy,ypred], feeddict_new)			
	feeddict_new = {x_holder: xval, y_holder:yval, train_holder: False}
	probs_val =  sess.run(probs, feeddict_new)			
	yret_val = (probs_val > 0.5)

	feeddict_new = {x_holder: xtest, y_holder:ytest, train_holder: False}
	probs_test =  sess.run(probs, feeddict_new)			
	yret_test = (probs_test > 0.5)
	print 'Val accuracy: ' , np.mean(yval == yret_val)
	
	
	sess.close()

ypos = np.array(np.where(yret_test == True))
yneg = np.array(np.where(yret_test == False))
print yneg.shape
print ypos.shape

plt.plot(xtest[ypos,0] , xtest[ypos,1], 's',color  = '#ffd0d0')
plt.plot(xtest[yneg,0], xtest[yneg,1], 's',color  = '#89fe05')
plt.axis([0, 10, 0, 10])


ypos = np.array(np.where(y == True))
yneg = np.array(np.where(y == False))
plt.plot(x[ypos,0] , x[ypos,1], 'o',color  = '#ff0000')
plt.plot(x[yneg,0], x[yneg,1], 'go')

#plt.title('28 x 28, dropout = 0.5')

plt.savefig("testSoft2_18_100_dropout.png")

plt.show()


	
#xlist = [column(wreturned1,0),column(wreturned2,0)]
#ylist = [column(wreturned1,1),column(wreturned2,1)]

#animateMultiple.animatePlots(xlist, ylist)

