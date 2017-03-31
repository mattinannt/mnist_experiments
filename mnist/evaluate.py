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

    height = img.shape[0]
    width = img.shape[1]

    max_dim = 300
    height_ar = int(float(height)/np.maximum(height, width) * max_dim)
    width_ar = int(float(width)/np.maximum(height, width) * max_dim)

    img = cv2.resize(img, (width_ar, height_ar), interpolation=cv2.INTER_AREA)

    # threshold image and find contours
    ret, thresh = cv2.threshold(img, 100, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]

    # sort bounding boxes from left to right
    rects_sorted = sorted(rects, key=lambda x: x[0])

    image_values = []
    for rect in rects_sorted:
        # slice image
        roi = thresh[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        height = roi.shape[0]
        width = roi.shape[1]

        height_ar = int(float(height)/np.maximum(height, width) * 24.0)
        width_ar = int(float(width)/np.maximum(height, width) * 24.0)

        roi = cv2.resize(roi, (width_ar, height_ar), interpolation=cv2.INTER_AREA)

        height_add_l = int(np.floor((28 - height_ar) / 2))
        width_add_b = int(np.floor((28 - width_ar) / 2))

        roi_padded = np.zeros([28, 28])

        for r in range(roi.shape[0]):
            for c in range(roi.shape[1]):
                roi_padded[r + height_add_l, c + width_add_b] = roi[r, c]

        roi = cv2.dilate(roi_padded, (3, 3))
        roi = roi/255.0  # scale to [0, 1]
        roi = roi > 0.2
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
