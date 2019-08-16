#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# 画像のあるディレクトリ##########################################

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = current_path + "/../data"
train_path = data_path + "/train"
train_img_dirs = os.listdir(train_path)
print(train_img_dirs)
box_df = pd.read_csv(data_path + "/box_list.csv", header=0)
NUM_CLASSES = len(box_df)  # 分類するクラス数
IMG_SIZE = 28 # 画像の1辺の長さ
print(box_df.name)


images = []
labels = []
for i, d in enumerate(box_df.name):
        # ./data/以下の各ディレクトリ内のファイル名取得
        files = os.listdir(train_path + "/" + d)
        for f in files:
            # 画像読み込み
            img = cv2.imread(train_path + "/"+ d + '/' + f)
            # 1辺がIMG_SIZEの正方形にリサイズ
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # 1列にして ####この処理はなくす
            img = img.flatten().astype(np.float32)/255.0
            images.append(img)
    
            # one_hot_vectorを作りラベルとして追加 ######変更
            #tmp = np.zeros(NUM_CLASSES)
            #tmp[i] = 1
            tmp = i
            labels.append(tmp)
# numpy配列に変換
images = np.asarray(images)
labels = np.asarray(labels)
# labels = keras.utils.to_categorical(labels, NUM_CLASSES)
# labels = labels[:1000]
print(len(images), len(labels))
# cv2.waitKey(10000)

# 短いシーケンシャルモデルを返す関数
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(IMG_SIZE*IMG_SIZE*3,)),#784
    #keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(IMG_SIZE,IMG_SIZE,3)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  return model

# 基本的なモデルのインスタンスを作る
model = create_model()
model.summary()

checkpoint_path = data_path + "/checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, verbose=1, mode='auto')
model = create_model()

# stratifiedkfold = StratifiedKFold(n_splits=4)

loss_datas = []
acc_datas =[]
test_images = []
test_labels = []


X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2 ,random_state=0, stratify=labels)
# for train_index, test_index in stratifiedkfold.split(images, labels):
model = create_model()
# history = model.fit(images[train_index], labels[train_index],  epochs = 50,
#           validation_data = (images[test_index],labels[test_index]),
#                     callbacks=[cp_callback])  # 訓練にコールバックを渡す
history = model.fit(X_train, y_train,  epochs = 50,
                    validation_data=(X_test, y_test),
                    callbacks=[cp_callback, early_stopping])  # 訓練にコールバックを渡す渡す
# loss,acc = model.evaluate(images[test_index],labels[test_index])
# test_images, test_labels = images[test_index] , labels[test_index]
# loss_datas.append(loss)
# acc_datas.append(acc)

# print(loss_datas)
# print(acc_datas)
# print(np.mean(acc_datas))


def plot_history(history):
    # print(history.history.keys())

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


# 学習履歴をプロット
plot_history(history)
