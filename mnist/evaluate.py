'''
restores trained model multi-layer from latest checkpoint file in path_to_trained_models


'''

import tensorflow as tf
import cv2
import numpy as np
import model_builder

def run(image_path):
    PATH_TO_MODELS = './models'
    PATH_TO_IMAGE = image_path


    # resize image and flatten
    img = cv2.imread(PATH_TO_IMAGE, 0)  # load as grayscale
    img = (img-255)/255.0  # scale to [0,1]
    img_resize = cv2.resize(img, (28, 28))
    img_resize_flat = img_resize.reshape([1, -1])

    # build model
    x = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y_conv = model_builder.cnn(x, keep_prob)

    # train & evaluate model
    saver = tf.train.Saver()
    with tf.Session() as sess:

        checkpoint_latest = tf.train.latest_checkpoint(PATH_TO_MODELS)

        print('restoring model '+checkpoint_latest)
        saver.restore(sess, checkpoint_latest)

        prediction_idx = sess.run(tf.argmax(y_conv, 1), feed_dict={x: img_resize_flat, keep_prob: 1.0})

        return prediction_idx[0]
