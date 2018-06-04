import numpy as np
import os
import re
import tensorflow as tf

DATA_DIR = "../training"

train_image = []

for i in range(50):
    train_image.append([])

base = []
pos = []
neg = []

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
            label = string.group(1)
            train_image[int(label) - 1].append(load_image(tmp_path))
        else:  
            load_data(tmp_path) 
            
def gen_pair():
    for i in range(1, 51):
        print(i)
        for j in range(0, 10):
            for k in range(0, 10):
                if j != k:
                    for l in range(1, 51):
                        if i != l:
                            for m in range(0, 10):
                                base.append([i - 1, j])
                                pos.append([i - 1, k])
                                neg.append([l - 1, m])

with tf.Session() as sess:                   
    load_data(DATA_DIR)
    gen_pair()

    base = np.array(base)
    pos = np.array(pos)
    neg = np.array(neg)

    train_image = np.array(train_image)

    np.save("base.npy", base)
    np.save("pos.npy", pos)
    np.save("neg.npy", neg)
    np.save("train_image.npy", train_image)