"""
solving mnist classification problem using tensorflow
multi-layer architecture
"""

# Config
BATCH_SIZE = 50
ITERATIONS = 20000

# Setup Logging
import logging
logging_format = '%(asctime)s - %(levelname)s - %(message)s'
log_level = logging.DEBUG
logging.basicConfig(filename='logfile.log',format=logging_format,level=log_level)
# create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(log_level)
# create formatter
formatter = logging.Formatter(logging_format)
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


logger.debug('STARTING MULTI-LAYER MNIST')


# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import TensorFlow and start interactive Session
import tensorflow as tf
session = tf.InteractiveSession()

# Create tf placeholders for input data and predictions
# x will be a 2d tensor with all images of the current batch * flattened pixel
# of the input image.
# y_ will be the probabilities for every image in the batch and every digit
# class
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# functionality to create weight-variables and bias-variables
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# reshape x
x_image = tf.reshape(x, [-1,28,28,1])
# convolve x_image
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely conntected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train & evaluate model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start time measurement
import time
start = time.time()

# initial logging
logger.debug('starting computation (batch-size: %d, iterations=%d)'%(BATCH_SIZE, ITERATIONS))

session.run(tf.global_variables_initializer())

for i in range(ITERATIONS):
  batch = mnist.train.next_batch(BATCH_SIZE)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    logging.debug("step %d, training accuracy %g"%(i, train_accuracy))

    time_elapsed = time.time() - start
    logger.debug('time elapsed: %.2fs'%(time_elapsed))
    logger.debug('mean seconds/batch: %fs'%(time_elapsed/(i+1)))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# stop time measurement
end = time.time()
computation_time = end - start

# print accuracy of test data & computation time
logger.debug("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
logger.debug('computation time: %.2fs'%(computation_time))
