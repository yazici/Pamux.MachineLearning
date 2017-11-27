

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================

# MNIST is a simple computer vision dataset
# It consists of images of handwritten digits
# It also includes labels for each image,
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py


"""A very simple MNIST classifier.
See extensive documentation at

https://www.tensorflow.org/get_started/mnist/beginners

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None
# The MNIST data is hosted on Yann LeCun's website.
# http://yann.lecun.com/exdb/mnist/

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


  #The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 
  #10,000 points of test data (mnist.test), 
  #and 5,000 points of validation data (mnist.validation). 
  # This split is very important: it's essential in machine learning that 
  # we have separate data which we don't learn from so that 
  # we can make sure that what we've learned actually generalizes!

  
  # As mentioned earlier, every MNIST data point has two parts: 
  # an image of a handwritten digit and a corresponding label. We'll call the images "x" and the labels "y".


  # A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.
  
  # If you want to assign probabilities to an object being one of several different things, 
  # softmax is the thing to do, because softmax gives us a list of values between 0 and 1 that 
  # add up to 1. Even later on, when we train more sophisticated models, the final step will be a layer of softmax.
  # A softmax regression has two steps: first we add up the evidence of our 
  # input being in certain classes, and then we convert that evidence into probabilities.

  # Softmax Intuition: http://neuralnetworksanddeeplearning.com/chap3.html#softmax

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  sess = tf.InteractiveSession()

  tf.global_variables_initializer().run()

  # Train

  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,

                                      y_: mnist.test.labels}))



if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',

                      help='Directory for storing input data')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)