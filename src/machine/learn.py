#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# データ水増し用
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
class Learn():
    def __init__(self, data_path):
       
        self.train_path = data_path + "/train"
        self.box_list_path = data_path + "/box_list.csv"
        
        self.checkpoint_path = data_path + "/checkpoint/cp.ckpt"
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        
        self.cp_callback = ModelCheckpoint(
            self.checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.early_stopping = EarlyStopping(
            monitor='val_loss', patience=20, verbose=1, mode='auto')

        self.img_size = (28, 28)
        self.box_df = pd.read_csv(self.box_list_path, header=0)
        self.class_num = len(self.box_df)

    def read_data(self):

        images = []
        labels = []
        for i , parts_name in enumerate(self.box_df.name):
            parts_dir_name = self.train_path + "/" + parts_name
            img_names = os.listdir(parts_dir_name)
            print(img_names)
            for f in img_names:
                img = cv2.imread(parts_dir_name + '/' + f)
                img = cv2.resize(img, self.img_size)
                images.append(img)
                labels.append(i)
        self.images = np.asarray(images).astype('float32')/255
        # self.images = self.images.astype('float32')/255.0
        self.labels = np_utils.to_categorical(labels, self.class_num)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(self.img_size[0], self.img_size[1], 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.class_num))
        model.add(Activation('softmax'))

        optimizers = "Adadelta"
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers, metrics=['accuracy'])
        model.summary()
        return model

    def learning(self, model):
        epochs = 200
        history = model.fit(self.images, self.labels,
                            validation_split=0.2, epochs=epochs, callbacks=[self.cp_callback, self.early_stopping])
        return history

def plot_history(history):
        # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

