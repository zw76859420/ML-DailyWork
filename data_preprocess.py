# -*- coding:utf-8 -*-
# author:zhangwei

import numpy as np
from keras.utils import np_utils

def data2one(data):
    """
       数据归一化；
    """
    # data = data_preprocess(filename)
    maxcols = data.max(axis=0)
    mincols = data.min(axis=0)
    data_shape = data.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows , data_cols))
    for i in range(data_cols):
        t[: , i] = (data[: , i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

def data_preprocess_cnn(filename):
    data = []
    labels = []
    with open(filename , 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.strip().split(',')
            labels.append(res[-1])
            data.append(res[:8])
    data = np.array(data , dtype=np.float)
    data = data2one(data)
    data = data.reshape(-1 , 1 , data.shape[1])
    # print(data.shape)
    labels = np.array(labels , dtype=np.int)
    labels = np_utils.to_categorical(labels, 2)
    return data , labels

def data_preprocess_dnn2(filename):
    data = []
    labels = []
    with open(filename , 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.strip().split(',')
            labels.append(res[-1])
            data.append(res[:8])
    data = np.array(data , dtype=np.float)
    data_1 = data2one(data)
    data = np.vstack((data , data_1))
    # data = data.reshape(-1 , 1 , data.shape[1])
    # print(data.shape)
    labels = np.array(labels , dtype=np.int)
    labels_1 = labels
    labels = np.hstack((labels , labels_1))
    labels = np_utils.to_categorical(labels, 2)
    return data , labels

def data_preprocess_dnn(filename):
    data = []
    labels = []
    with open(filename , 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.strip().split(',')
            labels.append(res[-1])
            data.append(res[:8])
    data = np.array(data , dtype=np.float)
    data_1 = data2one(data)
    # data = np.vstack((data , data_1))
    # data = data.reshape(-1 , 1 , data.shape[1])
    # print(data.shape)
    labels = np.array(labels , dtype=np.int)
    # labels_1 = labels
    # labels = np.hstack((labels , labels_1))
    labels = np_utils.to_categorical(labels, 2)
    return data , labels

if __name__ == '__main__':
    filename = 'data_set.txt'
    train_filename = '/home/zhangwei/PycharmProjects/ASR_MFCC/ml_work/train_set.txt'
    test_filename = '/home/zhangwei/PycharmProjects/ASR_MFCC/ml_work/test_set.txt'
    a , b = data_preprocess_dnn(train_filename)
    print(a.shape)