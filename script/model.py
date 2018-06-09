import numpy as np
import tensorflow as tf

from alexnet import AlexNet

SKIP_LAYER = ['fc8']

def cosine_distance(x1, x2):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
    
    x1_x2 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)

    # cosin = x1_x2 / (x1_norm * x2_norm)
    cosin = tf.divide(x1_x2, tf.multiply(x1_norm, x2_norm))
    return cosin

class Distinguisher:
    def __init__(self, base, inf, sess, reuse=False):
        DROP_PROB = 0.5
        CLASSES = 50

        base_embedding = AlexNet(base, DROP_PROB, CLASSES, SKIP_LAYER, sess, reuse=reuse).output
        inf_embedding = AlexNet(inf, DROP_PROB, CLASSES, SKIP_LAYER, sess, reuse=True).output
        feature = tf.concat([base_embedding, inf_embedding], 1)

        with tf.variable_scope('relation_network', reuse=reuse) as scope:
            print(feature.shape)
            # kernel = weight_variable([3, 3, 256, 64])
            # bias = bias_variable([64])
            # conv = tf.nn.conv2d(feature, kernel, strides=[1,2,2,1], padding='SAME')
            # conv1 = tf.nn.bias_add(conv, bias)
            # norm = tf.layers.batch_normalization(conv, reuse=reuse)
            # relu = tf.nn.relu(norm)
            # pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            # flat = tf.reshape(pool, [-1, 3*2*64])
            flat = tf.reshape(feature, [-1, 8192])
            fc = tf.layers.dense(inputs=flat, units=8, activation=tf.nn.relu, reuse=reuse, name='fc1')
            dense = tf.layers.dense(inputs=fc, units=1, activation=tf.nn.sigmoid, reuse=reuse, name='fc2')

        self.sim = dense
        
        tf.initialize_variables(tf.global_variables(scope='relation_network')).run()

        self.saver = tf.train.Saver()


def weight_variable(shape):  # you can use this func to build new variables
    # initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable('w', shape)
    #, initializer=tf.truncated_normal_initializer


def bias_variable(shape):  # you can use this func to build new variables
    # initial = tf.constant(0.1, shape=shape)
    return tf.get_variable('b', shape)
    #initializer=tf.constant_initializer(0.1)



