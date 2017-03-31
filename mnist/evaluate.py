'''
restores trained model multi-layer from latest checkpoint file in
path_to_trained_models
'''

import tensorflow as tf
import cv2
import numpy as np
from mnist import model_builder
import pdb


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

    # load image as grayscale, blur, and invert
    img = cv2.imread(PATH_TO_IMAGE, 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.bitwise_not(img)  # invert

    # threshold image and find contours
    ret, thresh = cv2.threshold(img, 50, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]

    # sort bounding boxes from left to right
    rects_sorted = sorted(rects, key=lambda x: x[0])

    image_values = []
    for rect in rects_sorted:
        # slice image
        # TODO: slice in the order of rect[0] to ensure that prediction have same order as in image
        roi = thresh[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi = roi/255.0  # scale to [0, 1]
        image_values.append(roi.reshape(-1))

    return run(np.vstack(image_values), model)


def run(image_values, model):

    x = model['x']
    y_conv = model['y_conv']
    keep_prob = model['keep_prob']
    sess = model['sess']

    prediction_idx, confidence = sess.run(
        [tf.argmax(y_conv, 1), tf.nn.softmax(y_conv)],
        feed_dict={x: image_values, keep_prob: 1.0})

    return prediction_idx, np.array([confidence[tuple] for tuple in zip(range(prediction_idx.shape[0]), prediction_idx)])
