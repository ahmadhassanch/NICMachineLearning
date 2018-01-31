# Import `tensorflow`
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result, Tensor("Mul:0", shape=(4,), dtype=int32)
print(result)   # lazy evaluation

# Evaluation Method 1
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run([result]))
sess.close() # Close the session

# Evaluation Method 2
sess = tf.Session()
print(sess.run(result))  # Print the result
sess.close() # Close the session

# Evaluation Method 3
with tf.Session() as sess:
  output = sess.run(result)
  print(output)

