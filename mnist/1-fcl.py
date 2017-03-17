"""
solving mnist classification problem using tensorflow
one fully connected layer
"""

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

# Define Model parameters as variables and initialize them with zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
session.run(tf.global_variables_initializer())

# Regression model (without softmax)
y = tf.matmul(x,W) + b

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# evaluate model
# get vector of booleans if prediction was correct
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# calculate accuracy by converting booleans to floats (0.0 || 1.0) and calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print accuracy of test-data
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
