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

# 画像のあるディレクトリ##########################################
train_img_dirs = os.listdir("../data")
print(train_img_dirs)
box_df = pd.read_csv("box_list.csv", header=0)
NUM_CLASSES = len(box_df)  # 分類するクラス数
IMG_SIZE = 28 # 画像の1辺の長さ

images = []
labels = []
for i, d in enumerate(train_img_dirs):
        # ./data/以下の各ディレクトリ内のファイル名取得
        files = os.listdir('../data/' + d)
        for f in files:
            # 画像読み込み
            img = cv2.imread('../data/' + d + '/' + f)
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

labels = labels[:1000]
print(len(images), len(labels))

#train_images = train_images[:1000].reshape(-1, IMG_SIZE * IMG_SIZE) / 255.0
#test_images = test_images[:1000].reshape(-1, IMG_SIZE * IMG_SIZE) / 255.0
#print(len(train_images), len(train_labels))


# In[37]:


# from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__
import matplotlib.pyplot as plt


# In[38]:


# 短いシーケンシャルモデルを返す関数
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(IMG_SIZE*IMG_SIZE*3,)),#784
    #keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(IMG_SIZE,IMG_SIZE,3)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  return model

# 基本的なモデルのインスタンスを作る
model = create_model()
model.summary()


# In[58]:


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

stratifiedkfold = StratifiedKFold(n_splits=3)

loss_datas = []
acc_datas =[]
test_images = []
test_labels = []

for train_index, test_index in stratifiedkfold.split(images, labels):
    model = create_model()
    model.fit(images[train_index], labels[train_index],  epochs = 50,#10
              validation_data = (images[test_index],labels[test_index]),
              callbacks = [cp_callback])  # 訓練にコールバックを渡す
    loss,acc = model.evaluate(images[test_index],labels[test_index])
    test_images, test_labels = images[test_index] , labels[test_index]
    loss_datas.append(loss)
    acc_datas.append(acc)


# In[62]:


print(loss_datas)


# In[60]:


print(acc_datas)


# In[61]:


print(np.mean(acc_datas))


# In[42]:


# get_ipython().system(u'ls {checkpoint_dir}')

