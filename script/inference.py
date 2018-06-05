import tensorflow as tf
import numpy as np
from model import Distinguisher
from DataLoader import load_image
import os
import re

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")

def inf_data(sess, model, path, train_dir):  
    for file in os.listdir(path):  
        tmp_path = os.path.join(path, file)  
        if not os.path.isdir(tmp_path):  
            # string = re.match(r'.*0([0-9]+)_0*([0-9]+)\.jpg', file)
            # label = string.group(1)
            print(tmp_path, inf_image(sess, model, tmp_path, train_dir))
        else:  
            inf_data(sess, model, tmp_path, train_dir) 

def inf_image(sess, model, pic_path, train_dir):
    sim = model.sim

    inf = load_image(pic_path) # 227 * 227 * 3
    inf = np.expand_dims(inf, axis=0)
    inf = inf.repeat(10, axis=0)

    score = np.zeros([50])
    for i in range(50):
        feed = { X_base: data[i], X_inf: inf }
        res = sess.run([sim], feed_dict=feed)

        # print(res[0])
        # print(np.asarray(res).shape)
        score[i] = np.sum(res[0])
            
    return np.argmax(score) + 1

test_path = '../testing'
with tf.Session() as sess:
    data = np.load('train_image.npy')

    _, _, HEIGHT, WIDTH, CHANNEL = data.shape
    
    X_base = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    X_inf = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])

    
    model = Distinguisher(X_base, X_inf, sess)

    model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))

    inf_data(sess, model, test_path, FLAGS.train_dir)



            