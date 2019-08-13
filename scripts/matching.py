#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
#from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
# import keras
import time

import rospy
from geometry_msgs.msg import Vector3
from Object import RecognizedObject

def cleanFiles():
    xml_dirs = os.listdir("./AR/xmls")
    img_dirs = os.listdir("./AR/imgs")
    if len(xml_dirs) > 0:
        for f in xml_dirs:
            print("remove:", f)
            os.remove('./AR/xmls/'+f)
    else:
        print("no pre files")
    if len(img_dirs) > 0:
        for f in img_dirs:
            print("remove:", f)
            os.remove('./AR/imgs/'+f)
    else:
        print("no pre files")    


if __name__ == "__main__":
    #tensorflow準備
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # train_img_dirs = os.listdir("../data")
    # print(train_img_dirs)
    # NUM_CLASSES = len(train_img_dirs) # 分類するクラス数
    #print(NUM_CLASSES)
    IMG_SIZE = 28 # 画像の1辺の長さ
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(IMG_SIZE*IMG_SIZE*3,)),#784
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
    model.load_weights(checkpoint_path)

    #前回ファイルの消去#########################################################
    #cleanFiles()
    
    #rosPubliserの準備
    rospy.init_node('matcher', anonymous=True)
    pub = rospy.Publisher('toCommunicator',Vector3,queue_size=10)
    
    while not rospy.is_shutdown():#画像分類の待機
        #画像データの格納
        objs = []
        images = []

        xml_dirs = os.listdir("./AR/xmls")
        if len(xml_dirs) > 0:
            #オブジェクトデータの取得
            for f in xml_dirs:
                obj = RecognizedObject('./AR/xmls/' + f)
                images.append(obj.image_np)
                objs.append(obj)
            images = np.asarray(images)
            predictions = model.predict(images)
            
            max_array = predictions.max(axis=1)
            max_index = predictions.argmax(axis=1)
            
            print(max_array, max_index)
            use_obj = max_array.argmax()    
            print(use_obj)
            #print(predictions)

            #送信用データの定義
            qu = Vector3()
            qu.x = objs[use_obj].x_m
            qu.y = objs[use_obj].y_m
            qu.z = max_index[use_obj]
            pub.publish(qu)
            #画像ファイルの消去###########################################################
            #cleanFiles()


        else:
            print("画像データ待機中")
            time.sleep(1)
    
    cv2.destroyAllWindows()
    print("complete")
