# 11/26/2017
# https://www.tensorflow.org/get_started/get_started

from __future__ import print_function


# __future__ is a real module, and serves three purposes:
# To avoid confusing existing tools that analyze import statements and expect to find the modules they’re importing.
# To ensure that future statements run under releases prior to 2.1 at least yield runtime exceptions (the import of __future__ will fail, because there was no module of that name prior to 2.1).
# To document when incompatible changes were introduced, and when they will be — or were — made mandatory. 
# # This is a form of executable documentation, and can be inspected programmatically via importing __future__ and examining its contents.


# Tensor: N-Dimensional Array of values


# 3 # a rank 0 tensor; a scalar with shape []
# [1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

# Rank of a tensor: # of dimensions?

import tensorflow as tf


# 1 Building the computational graph.
# 2 Running the computational graph.

# A computational graph is a series of TensorFlow operations arranged into a graph of nodes. 
#   Each node takes zero or more tensors as inputs and produces a tensor as an output. 
#   One type of node is a constant. 

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

print(node1, node2)  # Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

# Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect. 
# Instead, they are nodes that, when evaluated, would produce 3.0 and 4.0, respectively. 
# To actually evaluate the nodes, we must run the computational graph within a session. 
# A session encapsulates the control and state of the TensorFlow runtime.


sess = tf.Session()
print(sess.run([node1, node2])) # [3.0, 4.0]

#We can build more complicated computations by combining Tensor nodes with operations (Operations are also nodes). 
# For example, we can add our two constant nodes and produce a new graph as follows:


node3 = tf.add(node1, node2)
print("node3:", node3)   # node3: Tensor("Add:0", shape=(), dtype=float32)
print("sess.run(node3):", sess.run(node3)) # sess.run(node3): 7.0



# TensorFlow provides a utility called TensorBoard that can display a picture of the computational graph. 
 
# A graph can be parameterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# We can evaluate this graph with multiple inputs by using the feed_dict argument to the run method to feed concrete values to the placeholders:

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


# Variables allow us to add trainable parameters to a graph.

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# variables are initialized by the following call
init = tf.global_variables_initializer()  # This is creating a graph node, not running it yet
sess.run(init) # This runs the node.

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))  # [ 0.          0.30000001  0.60000002  0.90000004]


# assignment to W, and b variables.
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

 # TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function.
 #  The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss with respect to that variable.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)



sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))  # [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]