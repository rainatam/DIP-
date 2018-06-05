import os
import re
from PIL import Image
import numpy as np
import tensorflow as tf
import os
# import alexnet
from model import Distinguisher
# import AlexNet
# import glob
# import matplotlib.pyplot as plt

EPOCH = 100
EPISODE = 100
WAY = 50
SHOT = 10
DROP_PROB = 0.5
CLASSES = 50
BATCH_SIZE = 32
# QUERY = 
# SAMPLE = 

base = None
pos = None
neg = None
data = None
 
# saver = tf.train.Saver()

def train(sess, train_op, acc_op, y_pos, y_neg):
    assert(len(base) == len(pos))
    assert(len(base) == len(neg))
    length = len(base)

    st, ed, times = 0, BATCH_SIZE, 0
    iter = 0
    while st < length and ed <= length:
        print(iter)
        iter += 1
        a_base = []
        a_pos = []
        a_neg = []
        for i in range(st, ed):
            a_base.append(data[base[i][0]][base[i][1]].tolist())
            a_pos.append(data[pos[i][0]][pos[i][1]].tolist())
            a_neg.append(data[neg[i][0]][neg[i][1]].tolist())
        
        feed = {X_base: a_base, X_pos: a_pos, X_neg: a_neg}

        _, acc, _pos, _neg = sess.run([train_op, acc_op, y_pos, y_neg], feed_dict=feed)
        
        print("ACC: ", acc)
        # print(_pos)
        # print(_neg)

        st, ed = ed, ed + BATCH_SIZE
        times += 1

with tf.Session() as sess:

    learning_rate = 0.00001

    data = np.load('train_image.npy')
    base = np.load('base.npy')
    pos = np.load('pos.npy')
    neg = np.load('neg.npy')

    # print(HEIGHT, WIDTH, CHANNEL)

    print(data.shape)
    _, _, HEIGHT, WIDTH, CHANNEL = data.shape

    X_base = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    X_pos = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    X_neg = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    
    y_pos = Distinguisher(X_base, X_pos, sess).sim
    y_neg = Distinguisher(X_base, X_neg, sess, True).sim
    
    acc_op = tf.reduce_mean(tf.cast(tf.greater(y_pos, y_neg), tf.float32))

    loss = 1 - y_pos + y_neg
        
    prev = set(tf.global_variables()) 

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tf.trainable_variables())

    # tf.global_variables_initializer().run()
    tf.initialize_variables(list(set(tf.global_variables()) - prev)).run()
    # sess.run(init_new_vars_op)
    
    # print(list(set(tf.global_variables()) - prev))
    
    train(sess, train_op, acc_op, y_pos, y_neg)

    # save_path = saver.save(sess, '/train_image.npy')



        