"""
solving mnist classification problem using tensorflow
multi-layer architecture
"""

from mnist import model_builder
import time


def run():
    # Config
    BATCH_SIZE = 50
    ITERATIONS = 2000
    PATH_TO_MODELS = './mnist/models'

    import os
    if not os.path.exists(PATH_TO_MODELS):
        os.mkdir(PATH_TO_MODELS)

    # Setup Logging
    import logging
    logging_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_level = logging.DEBUG
    logging.basicConfig(filename='logfile.log', format=logging_format, level=log_level)
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
    # session = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # build model
    y_conv = model_builder.cnn(x, keep_prob)

    # train & evaluate model
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start time measurement
    start = time.time()

    # initial logging
    logger.debug('starting computation (batch-size: %d, iterations=%d)' % (BATCH_SIZE, ITERATIONS))

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:

        model_save_name = time.strftime('date_%d-%m-%Y_time_%H-%M-%S')
        sess.run(tf.global_variables_initializer())

        for i in range(ITERATIONS):
            batch = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                logging.debug("step %d, training accuracy %g" % (i, train_accuracy))

                time_elapsed = time.time() - start
                logger.debug('time elapsed: %.2fs' % (time_elapsed))
                logger.debug('mean seconds/batch: %fs' % (time_elapsed/(i+1)))

                # save model
                saver.save(sess, PATH_TO_MODELS+'/model_'+model_save_name+'.ckpt', i)

        # stop time measurement
        end = time.time()
        computation_time = end - start

        # print accuracy of test data & computation time
        logger.debug("test accuracy %g" % sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        logger.debug('computation time: %.2fs' % (computation_time))
