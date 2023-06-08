import tflearn
import tflearn.activations as activations
# Data loading and preprocessing
import tflearn.datasets.oxflower17 as oxflower17
from tflearn.activations import relu
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.normalization import local_response_normalization
from tflearn.utils import repeat
import glob
import cv2
import numpy as np
import os
import csv
import pandas as pd
from random import shuffle
from tqdm import tqdm
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
#----------------------------------#
def construct_inceptionv1onfire (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = input_data(shape=[None, y, x, 3])

    conv1_7_7 = conv_2d(network, 64, 5, strides=2, activation='relu', name = 'conv1_7_7_s2')

    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)

    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 128,3, activation='relu', name='conv2_3_3')

    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

    pool5_7_7 = avg_pool_2d(inception_4a_output, kernel_size=5, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)
    loss = fully_connected(pool5_7_7, 6,activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(loss, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)
    else:
        network = loss;

    model = tflearn.DNN(network, checkpoint_path='inceptiononv1onfire',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model
# ============================================#
labels = {
    0: 'Basketball',
    1: 'Football',
    2: 'Rowing',
    3: 'Swimming',
    4: 'Tennis',
    5: 'Yoga', }
TRAIN_DIR = 'Train'
TEST_DIR = 'Test'
IMG_SIZE = 150
LR = 0.001
MODEL_NAME = 'inception_resnet_v2'
# -----------------------------------------#
def create_label(image_name):
    """ Create a one-hot encoded vector from image name """
    word_label = image_name.split('_')[0]
    if word_label == 'Basketball':
        return np.array([1, 0, 0, 0, 0, 0])
    elif word_label == 'Football':
        return np.array([0, 1, 0, 0, 0, 0])
    elif word_label == 'Rowing':
        return np.array([0, 0, 1, 0, 0, 0])
    elif word_label == 'Swimming':
        return np.array([0, 0, 0, 1, 0, 0])
    elif word_label == 'Tennis':
        return np.array([0, 0, 0, 0, 1, 0])
    elif word_label == 'Yoga':
        return np.array([0, 0, 0, 0, 0, 1])


def create_data():
    data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img_data), create_label(img)])
    shuffle(data)
    return data


data = create_data()


# ============================================#
train = data
X_train = np.array([i[0] for i in train])
y_train = [i[1] for i in train]
#-------------------------------------------------------#
model = construct_inceptionv1onfire(IMG_SIZE, IMG_SIZE, training=True)


model.fit(X_train,y_train, n_epoch=10,
                    validation_set=0.2,
                    snapshot_step=200, show_metric=True, run_id=MODEL_NAME)
model.save('model.tfl')
pred_csv_data = []
for img in tqdm(os.listdir(TEST_DIR)):
    prediction = []
    path = os.path.join(TEST_DIR, img)
    img_ = cv2.imread(path)
    test_img = cv2.resize(img_, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(IMG_SIZE, IMG_SIZE,3)
    prediction = model.predict([test_img])[0]
    max_v = prediction[0]
    index = 0
    for i in range(1, len(prediction)):
        if prediction[i] > max_v:
            max_v = prediction[i]
            index = i
    pred_csv_data.append([img,index])


pred_csv_header = ['image_name', 'label']
with open('pred.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(pred_csv_header)
    # Use writerows() not writerow()
    writer.writerows(pred_csv_data)