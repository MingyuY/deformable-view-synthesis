#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:30:09 2018

@author: crazydemo
"""

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

image_height=64
image_width=64 

def data2fig(samples, img_size=64, nr=4, nc=4):
        #if self.is_tanh:
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(img_size, img_size, 3), cmap='Greys_r')
        return fig

def data2fig_4(samples, img_size=64, nr=4, nc=4):
        #if self.is_tanh:
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(img_size, img_size, 4) )
        return fig 

def read_and_decode_data(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),
                                                                    'p': tf.FixedLenFeature([], tf.int64),
                                                                    't': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    p = tf.cast(features['p'], tf.int32)
    t = tf.cast(features['t'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label, p, t  
