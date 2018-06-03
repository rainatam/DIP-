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

DATA_DIR = "../training"

train_image = []
train_label = []

def load_image(path):
    VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
    # load and preprocess the image
    img_string = tf.read_file(path)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_centered = tf.subtract(img_resized, VGG_MEAN)
    img_bgr = img_centered[:, :, ::-1]
    return img_bgr.eval()

def load_data(path):  
    for file in os.listdir(path):  
        tmp_path = os.path.join(path, file)  
        if not os.path.isdir(tmp_path):  
            string = re.match(r'.*0([0-9]+)_0*([0-9]+)\.jpg', file)
            train_image.append(load_image(tmp_path))
            train_label.append(string.group(1))
        else:  
            load_data(tmp_path) 

with tf.Session() as sess:
    load_data(DATA_DIR)
    train_image = np.array(train_image)
    train_label = np.array(train_label)
    _, HEIGHT, WIDTH, CHANNEL = train_image.shape


    X_train = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    X_test = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])

    embedding = AlexNet(X_train, DROP_PROB, CLASSES, [])

    y_train = tf.placeholder(tf.int64, [None])
    y_one_hot = tf.one_hot(y, depth = CLASSES)

    X_emb = embedding.fc7








# for i in range(EPOCH):
# 	for epi in range(EPISODE):
# 		epi_classes = np.random.permutation(ss)