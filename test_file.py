# -*- coding:utf-8 -*-
# author:zhangwei

filename = '/home/zhangwei/PycharmProjects/ASR_MFCC/ml_work/train_set.txt'
with open(filename , 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        print(line)