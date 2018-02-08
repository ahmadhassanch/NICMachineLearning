
data = tf.nn.batch_norm_with_global_normalization(...)

CONSTANTS
==========
z1 = tf.zeros([4,3])           #zeros
c1 = tf.constant([5,4,3])      #values, array

Uniform Distribution:
======================
uniFormDist = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)

VARIABLES
==========
Weights/bias/parameters, i.e., all tunable values are VARIABLES