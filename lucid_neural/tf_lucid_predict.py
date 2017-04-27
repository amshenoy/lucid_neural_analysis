#!/usr/bin/env python
import os
import tensorflow as tf
import numpy as np
from lucid_utils.classification.lucid_algorithm_data import classify
from lucid_utils.classification.lucid_algorithm import classify as classify2
from lucid_utils import data_api, blobbing
import urllib

def read(filename):
    frame = np.zeros((256, 256))
    f = (urllib.request.urlopen(filename).read()).decode("utf-8")
    lines = f.split("\n")
    for line in lines:
        if line != "":
                vals = line.split("\t")
                x = int(float(vals[0].strip()))
                y = int(float(vals[1].strip()))
                c = int(float(vals[2].strip()))
                frame[x][y] = c
    return frame

url = "http://starserver.thelangton.org.uk/lucid_dashboard/1932936732_0_0.txt"

## you can rename the classes but they must stay in the same order
classes = ["gamma", "beta", "muon", "proton", "alpha", "others"]
counts = {'alpha': 0, 'beta': 0, 'gamma': 0, 'others': 0, 'muon': 0, 'proton': 0}
counts2 = {'alpha': 0, 'beta': 0, 'gamma': 0, 'others': 0, 'muon': 0, 'proton': 0}
metrics = []
frame = read(url)

clusters = blobbing.find(frame, 1.5)
for cluster in clusters:
    metric = classify(cluster)
    metrics.append(metric)
    ## comparison with current algorithm
    counts2[classify2(cluster)] += 1

tf.reset_default_graph()

LOG_DIR = "./Model/"
model_name = 'model5.meta'

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

    opts = sess.run(predict_op, feed_dict={X: metrics})
    for i in range(len(opts)):
        counts[classes[opts[i]]] += 1
    print(counts)
    print(counts2)
