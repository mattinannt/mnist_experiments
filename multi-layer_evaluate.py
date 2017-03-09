'''
restores trained model multi-layer from latest checkpoint file in path_to_trained_models


'''

import tensorflow as tf
import cv2
import numpy as np
import sys

PATH_TO_MODELS = './models'
PATH_TO_IMAGE = sys.argv[1]


# resize image and flatten
img = cv2.imread(PATH_TO_IMAGE, 0)  # load as grayscale
img = (img-255)/255.0  # scale to [0,1]
img_resize = cv2.resize(img, (28, 28))
img_resize_flat = img_resize.reshape([1, -1])

# build_model
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

y_predicted = tf.argmax(y_conv, 1)

# train & evaluate model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
with tf.Session() as sess:

    checkpoint_latest = tf.train.latest_checkpoint(PATH_TO_MODELS)

    print('restoring model '+checkpoint_latest)
    saver.restore(sess, checkpoint_latest)

    prediction_idx = sess.run(y_predicted, feed_dict={x: img_resize_flat, keep_prob: 1.0})

    print(prediction_idx + 1)
