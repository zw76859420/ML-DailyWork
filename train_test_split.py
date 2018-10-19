# -*- coding:utf-8 -*-
# author:zhangwei

import numpy as np

filename = 'data_set.txt'
train_file = '/home/zhangwei/train_set.txt'
test_file = '/home/zhangwei/test_set.txt'

a = []
with open(filename , 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        a.append(line)

with open(train_file , 'w') as fwt:
     with open(test_file , 'w') as fwtf:
        n = len(a)
        j = 0
        for i in a:
            if j < n * 0.6:
                j += 1
                fwt.write(i)
            else:
                fwtf.write(i)