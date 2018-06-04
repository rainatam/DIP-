import os
import re
from PIL import Image
import numpy as np
import tensorflow as tf
import os
# import alexnet
from alexnet import *
# import AlexNet
# import glob
# import matplotlib.pyplot as plt

EPOCH = 100
EPISODE = 100
WAY = 50
SHOT = 10
DROP_PROB = 0.5
CLASSES  = 50
# QUERY = 
# SAMPLE = 

def gen_batch(base, pos, neg, data):

    return x_base, x_pos, x_neg

with tf.Session() as sess:
    learning_rate = 0.01

    train_image = np.load('train_image.npy')
    base = np.load('base.npy')
    pos = np.load('pos.npy')
    neg = np.load('neg.npy')

    # print(HEIGHT, WIDTH, CHANNEL)

    print(train_image.shape)
    _, _, HEIGHT, WIDTH, CHANNEL = train_image.shape

    X_base = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    X_pos = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    X_neg = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])

    with tf.variable_scope("PAIR", reuse=tf.AUTO_REUSE):
        X_base_embedding = AlexNet(X_base, DROP_PROB, CLASSES, []).output
        X_pos_embedding = AlexNet(X_pos, DROP_PROB, CLASSES, []).output
        X_neg_embedding = AlexNet(X_neg, DROP_PROB, CLASSES, []).output

    print(X_base_embedding.shape)

    y_pos = tf.losses.cosine_distance(X_base_embedding, X_pos_embedding, axis=1)
    y_neg = tf.losses.cosine_distance(X_base_embedding, X_neg_embedding, axis=1)

    loss = 1 - y_pos + y_neg

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tf.trainable_variables())

    # y_one_hot = tf.one_hot(y, depth = CLASSES)

    # X_emb = embedding.fc7








# for i in range(EPOCH):
# 	for epi in range(EPISODE):
# 		epi_classes = np.random.permutation(ss)