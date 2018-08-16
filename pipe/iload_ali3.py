# import pandas as pd

# csv_file = 'FILE23662'

# fileinfo = pd.read_csv(csv_file, header=None)

# label = fileinfo[1][0]

# fileinfo = fileinfo.drop([0,1],axis=1)

# fileinfo = fileinfo.sort_values(by=[3])

# print("Label: ", label)
# print(fileinfo.head(10))
# split -l 200000 file.txt new   

import click
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


APIS = []
APIFILE = '/home/mour/MlDl/data/AliSEC/api.list'
FOLDER = '/home/mour/MlDl/data/AliSEC/summary/2summary'
SEQUENCE = 200

with open(APIFILE) as f:
    for line in f:
        APIS.append(line.strip('\n'))


def processsummary(filename):
    apinfo = []
    label = os.path.basename(os.path.dirname(os.path.abspath(filename))).split('summary')[0]
    with open(filename) as f:
        for apicall in f.readlines():
            apicalls = apicall.strip('\n').split(' ')
            # print(apicalls)
            api_call_count = apicalls[-4]
            api_call_name = apicalls[-3]
            api_call_name_index = APIS.index(api_call_name)
            api_call_tid = apicalls[-2]
            api_call_return_value = apicalls[-1]

            apinfo.append([api_call_count,api_call_name_index,api_call_return_value])

    # return_value归一化，不是1就是0
    #此处处理文件类型的合并[(thread1，(api,return_value),(api,return_value)),(thread2,(api,return_value))]

    # apinfo.sort(key=lambda x: x[3])
    # 在此处pad_sequences到一致
    # x_api_train = np.asarray(list(map(lambda x: np.asarray(x[:-1]) , apinfo[:300])))
    x_api_train = np.asarray(apinfo[:300])
    # x_api_train = tf.keras.preprocessing.sequence.pad_sequences(x_api_train,maxlen=SEQUENCE,padding='pre',truncating='pre',value = 0)

    y_api_train = np.asarray(label)

    return x_api_train, y_api_train

def processfile(filename):
    apinfo = []

    with open(filename) as f:
        for apicall in f.readlines():
            apicalls = apicall.strip('\n').split(',')
            label = apicalls[1]
            api = apicalls[2]
            api_index = APIS.index(api)           
            api_tid = apicalls[3]
            api_tid_return = apicalls[4]
            api_tid_index = int(apicalls[5])

            # [(api,0),(api,1),(),]
            api_call = [api_index,api_tid,api_tid_return,api_tid_index]

            apinfo.append(api_call)

    # return_value归一化，不是1就是0
    #此处处理文件类型的合并[(thread1，(api,return_value),(api,return_value)),(thread2,(api,return_value))]

    apinfo.sort(key=lambda x: x[3])
    # 在此处pad_sequences到一致
    x_api_train = np.asarray(list(map(lambda x: np.asarray(x[:-1]) , apinfo)))
    y_api_train = label

    return x_api_train, y_api_train


def itrain_pipe():
    # 是否可以用MutliProcess去读取
    x_train = []
    y_train = []

    for r, d, f in os.walk(FOLDER):

        if os.path.basename(r) == '0summary':
            # We only get 1/4 of normal file
            for item in f:
                if int(item.split('FILE')[1]) %12 == 0:
                    x,y = processsummary(os.path.join(r, item))
                    x_train.append(x)
                    y_train.append(y)

        for item in f:
            x,y = processsummary(os.path.join(r, item))
            # x,y = processfile(os.path.join(r, item))
            x_train.append(x)
            y_train.append(y)
    x = np.asarray(x_train)
    y = np.asarray(y_train)

    x = tf.keras.preprocessing.sequence.pad_sequences(x,maxlen=SEQUENCE,padding='pre',truncating='pre',value = 0)
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42 )
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42 )
    
    return x_train, y_train, x_test, y_test