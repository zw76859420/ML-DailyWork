# -*- coding:utf-8 -*-
# author:zhangwei

from keras.models import Model
from keras.layers import Dense , Input , Conv2D , MaxPooling2D , LSTM , Conv1D , MaxPool1D , GRU , GlobalAveragePooling1D
from keras.layers import Dropout , BatchNormalization , Activation , Flatten , Reshape , regularizers
from keras.optimizers import Adam , SGD , RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint , EarlyStopping , ReduceLROnPlateau

from sklearn.cross_validation import train_test_split
import numpy as np
from ml_work.data_preprocess import *

"""
   产生训练集与测试集；
"""

train_filename = '/home/zhangwei/PycharmProjects/ASR_MFCC/ml_work/train_set.txt'
test_filename = '/home/zhangwei/PycharmProjects/ASR_MFCC/ml_work/test_set.txt'
train_data , train_label = data_preprocess_dnn2(train_filename)
test_data , test_label = data_preprocess_dnn(test_filename)

"""
   生成神经网络分类框架；
"""

input_data = Input(shape=[8] , name='input')

# layer1 = Conv1D(filters=16 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(input_data)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)

# layer1 = Conv1D(filters=16 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer1)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)

# layer1 = MaxPool1D(pool_size=[1])(layer1)
#
# layer1 = Conv1D(filters=32 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer1)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)

# layer1 = Conv1D(filters=32 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer1)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)

# layer1 = MaxPool1D(pool_size=[1])(layer1)

# layer1 = Conv1D(filters=64 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer1)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)

# layer1 = Conv1D(filters=64 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer1)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)
#
# layer1 = MaxPool1D(pool_size=[1])(layer1)

# layer1 = Conv1D(filters=128 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer1)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)

# layer1 = Conv1D(filters=128 , kernel_size=[1] , padding='same', activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer1)
# layer1 = BatchNormalization()(layer1)
# layer1 = Dropout(rate=0.3)(layer1)
#
# layer1 = MaxPool1D(pool_size=[1])(layer1)
#
# layer1 = GlobalAveragePooling1D()(layer1)

# layer2 = GRU(units=128 , use_bias=True , activation='relu' , return_sequences=True , go_backwards=True)(layer1)
# layer2 = BatchNormalization()(layer2)
# layer2 = Dropout(rate=0.4)(layer2)
#
# layer3 = GRU(units=128 , use_bias=True , activation='relu' , return_sequences=True , go_backwards=True)(layer2)
# layer3 = BatchNormalization()(layer3)
# layer3 = Dropout(rate=0.4)(layer3)
#
# layer3 = GRU(units=128 , use_bias=True , activation='relu' , return_sequences=True , go_backwards=True)(layer3)
# layer3 = BatchNormalization()(layer3)
# layer3 = Dropout(rate=0.3)(layer3)

# reshape = Flatten()(layer1)
# layer4 = Dense(units=256 , use_bias=True , activation='relu' , kernel_initializer='he_normal')(layer1)
# layer4 = BatchNormalization()(layer4)
# layer4 = Dropout(rate=0.3)(layer4)

# layer4 = Dense(units=512 , use_bias=True , activation='relu' , kernel_initializer='he_normal')(layer4)
# layer4 = BatchNormalization()(layer4)
# layer4 = Dropout(rate=0.3)(layer4)

# layer4 = Dense(units=2 , use_bias=True)(layer4)
# prediction = Activation(activation='softmax')(layer4)
# model = Model(input_data , prediction)
# model.summary()

layer1 = Dense(units=512 , use_bias=True , kernel_initializer='he_normal' , activation='relu')(input_data)
layer1 = BatchNormalization()(layer1)
layer1 = Dropout(rate=0.4)(layer1)

layer2 = Dense(units=512 , use_bias=True , kernel_initializer='he_normal' , activation='relu')(layer1)
layer2 = BatchNormalization()(layer2)
layer2 = Dropout(rate=0.4)(layer2)

layer3 = Dense(units=1024, use_bias=True , kernel_initializer='he_normal' , activation='relu')(layer2)
layer3 = BatchNormalization()(layer3)
layer3 = Dropout(rate=0.4)(layer3)

layer4 = Dense(units=1024 , use_bias=True , kernel_initializer='he_normal' , activation='relu')(layer3)
layer4 = BatchNormalization()(layer4)
layer4 = Dropout(rate=0.4)(layer4)

layer5 = Dense(units=2)(layer4)
prediction = Activation(activation='softmax')(layer5)
model = Model(input_data , prediction)

adam = Adam(lr=0.01 , beta_1=0.9 , beta_2=0.999 , epsilon=1e-5)
rms = RMSprop(lr=0.01 , decay=0.9)

model.compile(optimizer=adam , loss='categorical_crossentropy' , metrics=['accuracy'])

checkpointer = [ModelCheckpoint(filepath='/home/zhangwei/model.h5' , verbose=1 , save_best_only=True)]

model.fit(x=train_data , y=train_label , batch_size=128 , epochs=1000 , callbacks=checkpointer)

loss , evaluate = model.evaluate(x=test_data , y=test_label)
print('Test Accuracy : ' , evaluate)
