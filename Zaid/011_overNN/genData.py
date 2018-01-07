import numpy as np 
import tensorflow as tf


input_length = 2		# this is a constant for this example
dataset_size = 100000
num_classes = 1
hidden1_size = 6
hidden2_size = 6

print "Hello"
scale = 2
bias = 0.5


x = np.random.rand(dataset_size,input_length) - bias # matrix of inputs

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

#y = logits > 0.
y = np.argmax(logits,axis = 1)
print y.shape

print np.mean(y==True)