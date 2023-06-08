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

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

conv1 = conv_2d(conv_input, 32, 3, activation='relu')
pool1 = max_pool_2d(conv1, 3)

conv2 = conv_2d(pool1, 64, 3, activation='relu')
pool2 = max_pool_2d(conv2, 3)

conv3 = conv_2d(pool2, 128, 3, activation='relu')
pool3 = max_pool_2d(conv3, 3)

conv4 = conv_2d(pool3, 32, 3, activation='relu')
pool4 = max_pool_2d(conv4, 3)

conv5 = conv_2d(pool4, 64, 3, activation='relu')
pool5 = max_pool_2d(conv5, 3)

fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=7)

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=45,
                    validation_set=({'input': X_test}, {'targets': y_test}),
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
with open('SC_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(pred_csv_header)
    # Use writerows() not writerow()
    writer.writerows(pred_csv_data)