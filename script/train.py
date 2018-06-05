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

def train(sess, model, train_op, acc_op, loss_op, train_dir):
    assert(len(base) == len(pos))
    assert(len(base) == len(neg))
    length = len(base)

    st, ed, times = 0, BATCH_SIZE, 0
    iter = 0
    sum_acc = 0
    sum_loss = 0
    while st < length and ed <= length:
        a_base = []
        a_pos = []
        a_neg = []
        for i in range(st, ed):
            a_base.append(data[base[i][0]][base[i][1]].tolist())
            a_pos.append(data[pos[i][0]][pos[i][1]].tolist())
            a_neg.append(data[neg[i][0]][neg[i][1]].tolist())
        
        feed = {X_base: a_base, X_pos: a_pos, X_neg: a_neg}

        _, acc, loss = sess.run([train_op, acc_op, loss_op], feed_dict=feed)

        sum_acc += acc
        sum_loss += loss

        iter += 1

        if iter % 10 == 0:
            print("iter: %d acc: %f loss: %f" % (iter, sum_acc/10, sum_loss/10))
            sum_acc = 0
            sum_loss = 0

        

        model.saver.save(sess, '%s/model' % train_dir)

        # print(_pos)
        # print(_neg)

        st, ed = ed, ed + BATCH_SIZE
        times += 1
        
        
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")

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
    
    # y_pos = Distinguisher(X_base, X_pos, sess).sim
    model = Distinguisher(X_base, X_pos, sess)
    y_pos = model.sim
    y_neg = Distinguisher(X_base, X_neg, sess, True).sim
    
    acc_op = tf.reduce_mean(tf.cast(tf.greater(y_pos, y_neg), tf.float32))

    loss = 1 - y_pos + y_neg
    
    loss_op = tf.reduce_mean(loss)

    if tf.train.get_checkpoint_state(FLAGS.train_dir):
        print("Reading model parameters from %s" % FLAGS.train_dir)
        model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
    else:
        print("Created model with fresh parameters.")
        
    prev = set(tf.global_variables()) 
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tf.trainable_variables())
    # tf.global_variables_initializer().run()
    tf.initialize_variables(list(set(tf.global_variables()) - prev)).run()
    # sess.run(init_new_vars_op)
    
    # print(list(set(tf.global_variables()) - prev))
    
    train(sess, model, train_op, acc_op, loss_op, FLAGS.train_dir)

    # save_path = saver.save(sess, '/train_image.npy')



        