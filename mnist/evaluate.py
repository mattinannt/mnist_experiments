'''
restores trained model multi-layer from latest checkpoint file in
path_to_trained_models
'''

import tensorflow as tf
import cv2
import numpy as np
from mnist import model_builder


def init():
    PATH_TO_MODELS = './mnist/models/'

    # build model
    x = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y_conv = model_builder.cnn(x, keep_prob)

    # train & evaluate model
    saver = tf.train.Saver()
    sess = tf.Session()

    checkpoint_latest = tf.train.latest_checkpoint(PATH_TO_MODELS)

    print('restoring model ' + checkpoint_latest)

    saver.restore(sess, checkpoint_latest)

    return {'x': x, 'y_conv': y_conv, 'keep_prob': keep_prob, 'sess': sess}


def from_local_image(image_path, model):

    PATH_TO_IMAGE = image_path

    # resize image and flatten
    img = cv2.imread(PATH_TO_IMAGE, 0).astype(int)  # load as grayscale
    img = (255-img)/255.0  # scale to [0,1] and invert
    img_resize = cv2.resize(img, (28, 28))
    image_values = img_resize.reshape([1, -1])

    return run(image_values, model)


def run(image_values, model):

    x = model['x']
    y_conv = model['y_conv']
    keep_prob = model['keep_prob']
    sess = model['sess']

    prediction_idx, confidence = sess.run(
        [tf.argmax(y_conv, 1), tf.nn.softmax(y_conv)],
        feed_dict={x: image_values, keep_prob: 1.0})

    return prediction_idx[0], confidence[0][prediction_idx[0]]
