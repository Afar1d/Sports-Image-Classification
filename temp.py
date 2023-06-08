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
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
from tflearn.layers.normalization import local_response_normalization
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
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'Sport-cnn'
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
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE),3)
        data.append([np.array(img_data), create_label(img)])
    shuffle(data)
    return data


data = create_data()

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# ============================================#
train = train_data
test = test_data
X_train = np.array([i[0] for i in train])
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test])
y_test = [i[1] for i in test]

# -------------------------------------------------------#
from tflearn.layers.normalization import batch_normalization
net = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

net = conv_2d(net, 96, 7, strides=2, activation='relu')

net = batch_normalization(net)
net = max_pool_2d(net, 2)
net = dropout(net, 0.8)

net = conv_2d(net, 256, 5, strides=2, activation='relu')
net = batch_normalization(net)

net = max_pool_2d(net, 2)
net = dropout(net, 0.8)

net = conv_2d(net, 384, 3, activation='relu')
net = conv_2d(net, 384, 3, activation='relu')
net = conv_2d(net, 256, 3, activation='relu')
net = batch_normalization(net)
net = max_pool_2d(net, 2)
net = dropout(net, 0.8)

net = fully_connected(net, 4096, activation='tanh')
net = dropout(net, 0.5)
net = fully_connected(net, 4096, activation='tanh')
net = dropout(net, 0.5)
net = fully_connected(net, 6, activation='softmax')
net = regression(net, optimizer='adam',
                 loss='categorical_crossentropy', learning_rate=0.0001)
model = tflearn.DNN(net, tensorboard_dir='log', tensorboard_verbose=0)


history = model.fit(X_train,y_train, n_epoch=15,
                    validation_set=(X_test, y_test),
                    snapshot_step=200, show_metric=True, run_id=MODEL_NAME)
model.save('model.tfl')


pred_csv_data = []
for img in tqdm(os.listdir(TEST_DIR)):
    prediction = []
    path = os.path.join(TEST_DIR, img)
    img_ = cv2.imread(path)
    test_img = cv2.resize(img_, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
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