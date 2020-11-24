#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:41:44 2019

@author: ymy
"""
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from itertools import permutations
import tensorflow as tf
from glob import glob
import pandas as pd
import argparse
import time

def creat_dataset(path):
    img_dir = path 
    info_dic =  img_dir+'/all_chair_names.mat'
    import scipy.io as scio 
    info_mat = scio.loadmat(info_dic) 
    folder_names = info_mat['folder_names'][0]
    order = np.arange(len(folder_names))
#    np.random.shuffle(order)
    train_num = order[:int(len(folder_names)*0.8)+1]
    test_num = order[int(len(folder_names)*0.8)+1:]
    train_name = folder_names[train_num]
    test_name = folder_names[test_num]
    with open('datasets/dataset.txt','w') as f:
        for ii, i in enumerate(train_name):
            img_list = glob(img_dir+'/' + str(i[0]) + '/*/*.png')
            for j in range(len(img_list)):
                str_ = img_list[j].split('/')[-3] + '/' + img_list[j].split('/')[-2] + '/' + img_list[j].split('/')[-1]
                f.write(str_)
                f.write(' ')
                name = img_list[j].split('/')[-1]
                f.write(name.split('_')[1])
                f.write(' ') 
                f.write(name.split('_')[2])
                f.write(' ') 
                f.write(name.split('_')[3])
                f.write('\n')
        f.close()
 
    count = 0
    writer = tf.python_io.TFRecordWriter("datasets/dataset.tfrecords")
    with open('datasets/dataset.txt', 'r') as f:
        for line in f:
            line = line.strip()
            field = line.split(' ') 
            label = [int(field[1])]
            p = [int(field[2][1:])]
            t = [int(field[3][1:])]
            img = Image.open(path + '/' +field[0])
            if float(img.size[0])/float(img.size[1])>4 or float(img.size[1])/float(img.size[0])>4:
                continue
            img= img.resize((128,128))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),\
                                                                           'p': tf.train.Feature(int64_list=tf.train.Int64List(value=p)),\
                                                                           't': tf.train.Feature(int64_list=tf.train.Int64List(value=t)),\
                                                                           'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))

            writer.write(example.SerializeToString())
            count = count + 1 
            if count%500 ==0:
                print 'Time:{0},{1} images are processed.'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),count)
        print "%d images are processed." %count
    print 'Done!'
    writer.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='datasets path')
    args = parser.parse_args()
    creat_dataset(args.path)