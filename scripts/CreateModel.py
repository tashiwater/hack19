#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
# import numpy as np
import cv2
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
# import xml.etree.ElementTree as ET

# 画像のあるディレクトリ##########################################
# train_img_dirs = os.listdir("../data")
print(train_img_dirs)
box_df = pd.read_csv("box_list.csv", header=0)
NUM_CLASSES = len(box_df)  # 分類するクラス数
IMG_SIZE = 28 # 画像の1辺の長さ

images = []
labels = []
for i in range(NUM_CLASSES):
    # ./data/以下の各ディレクトリ内のファイル名取得
    files = os.listdir('../data/' + box_df.iloc[i].name)
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


# inputファイルを参照

# In[69]:


input_images = []
input_labels = []
test_img_dirs = os.listdir("../input_data")
for i, d in enumerate(test_img_dirs):
    # ./data/以下の各ディレクトリ内のファイル名取得
    files = os.listdir('../input_data/' + d)
    for f in files:
        # 画像読み込み
        img = cv2.imread('../input_data/' + d + '/' + f)
        # 1辺がIMG_SIZEの正方形にリサイズ
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # 1列にして ####この処理はなくす
        img = img.flatten().astype(np.float32)/255.0
        input_images.append(img)

        # one_hot_vectorを作りラベルとして追加 ######変更
        #tmp = np.zeros(NUM_CLASSES)
        #tmp[i] = 1
        tmp = i
        input_labels.append(tmp)
# numpy配列に変換
input_images = np.asarray(input_images)
input_labels = np.asarray(input_labels)

input_labels = input_labels[:1000]
print(len(input_images), len(input_labels))

test_images = input_images
test_labels = input_labels


# 学習データがない場合

# In[70]:


model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# 学習データがある場合

# In[71]:


model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[72]:


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(train_img_dirs[predicted_label],
                                    100*np.max(predictions_array),
                                    train_img_dirs[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# In[73]:


predictions = model.predict(test_images)
for i in range(len(predictions)):
    print("result: ",train_img_dirs[np.argmax(predictions[i])].rjust(6),"     answer:",train_img_dirs[test_labels[i]].rjust(6))


# In[74]:


test_images_fig = []
for i in range(len(test_images)):
    test_images_fig.append(np.reshape(test_images[i], (IMG_SIZE,IMG_SIZE,3)))
num_rows = NUM_CLASSES
num_cols = 1
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images_fig)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()


# In[68]:


plt.figure(figsize=(10,10))
np.set_printoptions(precision=0)
for i in range(len(test_images_fig)):
    plt.subplot((len(test_images_fig)/3)+1,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_fig[i], cmap=plt.cm.binary)
    #plt.xlabel(train_img_dirs[test_labels[i]])
    plt.xlabel("{}   {:.2f}%".format(train_img_dirs[np.argmax(predictions[i])], 
                                         100*np.amax(predictions[i]))
              )
plt.show()


# In[26]:


# ファイル名に(`str.format`を使って)エポック数を埋め込みます
#checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

#cp_callback = tf.keras.callbacks.ModelCheckpoint(
#    checkpoint_path, verbose=1, save_weights_only=True,
    # 重みを5エポックごとに保存します
#    period=5)

#model = create_model()
#model.fit(train_images, train_labels,
#          epochs = 100, callbacks = [cp_callback],
#          validation_data = (test_images,test_labels),
#          verbose=0)


