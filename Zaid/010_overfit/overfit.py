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
hidden1_size = 50
hidden2_size = 50


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

y1 = np.square(x[:,1] - 2) + np.square(x[:,0] - 2)	# line defining decision boundary
y2 = np.square(x[:,1] - 8) + np.square(x[:,0] - 8)
y3 = np.square(x[:,1] - 5) + np.square(x[:,0] - 5)
y4 = np.square(x[:,1] - 8) + np.square(x[:,0] - 1)
#y = np.random.rand(dataset_size,)
y = np.logical_or((y1 < 2),(y2 < 2))
y = np.logical_or(y,(y3 < 2))
y = np.logical_or(y,(y4 < 2))
y = y.reshape(dataset_size,1)

#y[rand_indices] = np.logical_not(y[rand_indices])




xval = np.fromfile('xval.csv',sep=',')	# matrix of inputs
print xval.shape
xval = xval.reshape(testset_size, input_length)


y1 = np.square(xval[:,1] - 2) + np.square(xval[:,0] - 2)	# line defining decision boundary
y2 = np.square(xval[:,1] - 8) + np.square(xval[:,0] - 8)
y3 = np.square(xval[:,1] - 5) + np.square(xval[:,0] - 5)
y4 = np.square(xval[:,1] - 8) + np.square(xval[:,0] - 1)

#y = np.random.rand(dataset_size,)
yval = np.logical_or((y1 < 4),(y2 < 4))
yval = np.logical_or(yval,(y3 < 4))
yval = np.logical_or(yval,(y4 < 4))
yval = yval.reshape(testset_size,1)

#rand_indices=[randint(0,9) for p in range(0,9)]
#yval[rand_indices] = np.logical_not(yval[rand_indices])




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
print 'xtest' ,xtest.shape
print  y.astype(int) 
print 'y shape', y.shape



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
	
	h1 = tf.nn.relu(tf.matmul(x_holder,w1) + b1)
	h1 = tf.layers.dropout(h1,rate = 0.5, training = train_holder)

	w2_shape = [hidden1_size, hidden2_size]
	w2 = tf.Variable(tf.truncated_normal(shape = w2_shape, stddev = 2.0/(hidden1_size+ hidden2_size)))
	b2 = tf.Variable(tf.truncated_normal(shape = [1,hidden2_size], stddev = 2.0/(hidden1_size+ hidden2_size)))

	h2 = tf.nn.relu(tf.matmul(h1,w2)) + b2
	h2 = tf.layers.dropout(h2,rate = 0.5, training = train_holder)

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
	for i in xrange(10000):
		for j in xrange(batches):
			feeddict = {x_holder: x, y_holder:y, train_holder: True}
			probs_ret,lret,_ ,acc=  sess.run([probs,loss,train_op,accuracy], feeddict)
		if (i+1) % 10 == 0 :
			feeddict_new = {x_holder: xval, y_holder:yval, train_holder: False}
			acc_test =  sess.run(accuracy, feeddict_new)		
			print (i+1),lret,acc,acc_test
			summary_str = sess.run(summary, feed_dict=feeddict)
			summary_writer.add_summary(summary_str,i)
			summary_writer.flush()


	yret = (probs_ret > 0.5)
	print 'train accuracy: ' , np.mean(y == yret) , acc
	
	#feeddict_new = {x_holder: xval, y_holder:yval, train_holder: False}
	#probs_val,pred,acc,pred_y =  sess.run([probs,correct_prediction,accuracy,ypred], feeddict_new)			
	feeddict_new = {x_holder: xval, y_holder:yval, train_holder: False}
	probs_val,acc =  sess.run([probs,accuracy], feeddict_new)			
	yret_val = (probs_val > 0.5)
	print 'Val accuracy: ' , np.mean(yval == yret_val),acc
	
	

	feeddict_new = {x_holder: xtest, y_holder:ytest, train_holder: False}
	probs_test =  sess.run(probs, feeddict_new)			
	yret_grid = (probs_test > 0.5)
	#print 'test accuracy: ' , np.mean(ytest == yret_test)
	#print 'trues', np.mean(True == yret_test)
	sess.close()

ypos_valid = np.array(np.where(yval      == True))
yneg_valid = np.array(np.where(yval      == False))
ypos_grid  = np.array(np.where(yret_grid == True))
yneg_grid  = np.array(np.where(yret_grid == False))
ypos = np.array(np.where(y == True))
yneg = np.array(np.where(y == False))
print yneg.shape
print ypos.shape

plt.subplot(2, 2, 1)
plt.plot(x[ypos,0] , x[ypos,1], 'o',color  = '#ff0000')
plt.plot(x[yneg,0], x[yneg,1], 'go')
plt.plot(xval[ypos_valid,0], xval[ypos_valid,1], '+',color  = '#ff0000')
plt.plot(xval[yneg_valid,0], xval[yneg_valid,1], 'g+')

plt.subplot(2, 2, 3)
plt.axis([0, 10, 0, 10])
plt.plot(x[ypos,0] , x[ypos,1], 'o',color  = '#ff0000')
plt.plot(x[yneg,0], x[yneg,1], 'go')

plt.subplot(2, 2, 4)
plt.axis([0, 10, 0, 10])
plt.plot(xtest[ypos_grid,0], xtest[ypos_grid,1], 's',color  = '#ffd0d0')
plt.plot(xtest[yneg_grid,0], xtest[yneg_grid,1], 's',color  = '#89fe05')
plt.plot(x[ypos,0] , x[ypos,1], 'o',color  = '#ff0000')
plt.plot(x[yneg,0], x[yneg,1], 'go')

#plt.title('28 x 28, dropout = 0.5')

plt.savefig( '%s%d%s' %("hidden 2 = ", hidden2_size,".png"))

plt.show()


	
#xlist = [column(wreturned1,0),column(wreturned2,0)]
#ylist = [column(wreturned1,1),column(wreturned2,1)]


