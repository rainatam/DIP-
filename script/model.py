import numpy as np
import tensorflow as tf

from alexnet import AlexNet

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

        base_embedding = AlexNet(base, DROP_PROB, CLASSES, ['fc7', 'fc8'], sess, reuse=reuse).output
        inf_embedding = AlexNet(inf, DROP_PROB, CLASSES, ['fc7', 'fc8'], sess, reuse=True).output

        self.sim = cosine_distance(base_embedding, inf_embedding)