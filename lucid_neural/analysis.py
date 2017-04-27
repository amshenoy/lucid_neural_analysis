#!/usr/bin/env python
import os
import tensorflow as tf
import numpy as np
from lucid_utils.classification.lucid_algorithm_data import classify

## you can rename the classes but they must stay in the same order
classes = ["gamma", "beta", "muon", "proton", "alpha", "others"]

def predict(cluster):
    tf.reset_default_graph()

    this_dir, this_filename = os.path.split(__file__)
    model_name = '\model5.meta'
    LOG_DIR = os.path.join(this_dir, "Model")

    checkpoint = tf.train.latest_checkpoint(LOG_DIR)
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.import_meta_graph(LOG_DIR+model_name)
        saver.restore(sess, checkpoint)
        X = tf.get_collection("X")[0]
        predict_op = tf.get_collection('predict_op')[0]
        full_op = tf.get_collection('full_op')[0]

        index = sess.run(predict_op, feed_dict={X: [classify(cluster)]})[0]
        return classes[index]
