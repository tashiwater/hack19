#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
# import time

import rospy
from geometry_msgs.msg import Vector3
from std_srvs.srv import SetBool, SetBoolResponse

from machine.Object import RecognizedObject
from machine.img_generator import padding
from machine.learn import Learn
class Match():
    def __init__(self, data_path, testdata_path, pub_topic, match_srv, get_target_srv):
        self.testdata_path = testdata_path
        learn = Learn(data_path)
        self.img_size = learn.img_size
        self.model = learn.create_model()
        self.model.load_weights(learn.checkpoint_path)
        while not rospy.is_shutdown() and self.match() is False:
            pass
        self.pub = rospy.Publisher(pub_topic, Vector3, queue_size=1)
        rospy.Service(match_srv, SetBool, self.srv_callback)
        rospy.Service(get_target_srv, SetBool, self.get_target_srv_callback)

    def srv_callback(self, request):
        resp = SetBoolResponse()
        resp.message = "called. data: " + str(request.data)
        print(resp.message)
        resp.success = self.match()
        return resp

    def get_target_srv_callback(self, request):
        resp = SetBoolResponse()
        
        resp.message = "called. data: " + str(request.data)
        print(resp.message)
        if self.i >= len(self.df):  # 全部出力したらfalseを返す
            resp.success = False
        else:
            resp.success = True
            self.pub_target()
        return resp

    def match(self):
        print("matching")
        if self.get_test_data() is False:
            return False
        self.predict()
        return True

    def pub_target(self):
        use_obj = self.df.index[self.i]
        print("target_img", use_obj)
        qu = Vector3()
        qu.x = self.objs[use_obj].x_m
        qu.y = self.objs[use_obj].y_m
        qu.z = self.max_index[use_obj]
        print("pub", qu)
        self.pub.publish(qu)
        self.i += 1
        
    def run(self):
        if self.i >= len(self.df):  # 全部出力したら再度画像を取得して予測
            self.match()

    def get_test_data(self):
        self.objs = []
        images = []

        xml_dirs = os.listdir(self.testdata_path+"/xmls")
        print(xml_dirs)
        if len(xml_dirs) < 1:
            print("there is no xml")
            return False
        #オブジェクトデータの取得
        for f in xml_dirs:
            obj = RecognizedObject(self.testdata_path+'/xmls/' + f)
            img = padding(obj.image)
            img = cv2.resize(img, self.img_size)
            images.append(img)
            self.objs.append(obj)
        self.images = np.asarray(images).astype('float32')/255
        return True
    
    def predict(self):
        predictions = self.model.predict(self.images)
        # print(predictions)
        max_array = predictions.max(axis=1)
        self.max_index = predictions.argmax(axis=1)
        print(max_array, self.max_index)
        self.i = 0
        self.df = pd.Series(max_array)
        self.df.sort_values(ascending=False, inplace=True)
   


"""
current_path = os.path.dirname(os.path.abspath(__file__))
testdata_path = current_path + "/../data/test"


if __name__ == "__main__":
    #tensorflow準備
    checkpoint_path = current_path+"/training_1/cp.ckpt"
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

        xml_dirs = os.listdir(testdata_path+"/xmls")
        if len(xml_dirs) > 0:
            #オブジェクトデータの取得
            for f in xml_dirs:
                obj = RecognizedObject(testdata_path+'/xmls/' + f)
                images.append(obj.image_np)
                objs.append(obj)
            images = np.asarray(images)
            predictions = model.predict(images)
            
            max_array = predictions.max(axis=1)
            max_index = predictions.argmax(axis=1)
            print(max_array, max_index)
            df = pd.Series(max_array)
            print(df)
            df.sort_values(ascending = False, inplace=True)
            print("sort", df)
            def generate_index():
                for i in df.index:
                    yield i
            gen = generate_index()
            #一時的にsrvを適当に使う。後でちゃんとクラスにするべき.srvで値が返せるようにする
            def callback(request):
                resp = SetBoolResponse()
                resp.success = True
                resp.message = "called. data: " + str(request.data)
                print(resp.message)
                use_obj = next(gen)
                print("target_img", use_obj)
                qu = Vector3()
                qu.x = objs[use_obj].x_m
                qu.y = objs[use_obj].y_m
                qu.z = max_index[use_obj]
                print("pub", qu)
                pub.publish(qu)
                return resp
            rospy.Service("match_srv", SetBool, callback)
            rospy.spin()
           
        #     use_obj = max_array.argmax()    
        #     print(use_obj)
        #     #print(predictions)

        #     #送信用データの定義
        #     qu = Vector3()
        #     qu.x = objs[use_obj].x_m
        #     qu.y = objs[use_obj].y_m
        #     qu.z = max_index[use_obj]
        #     pub.publish(qu)
        #     #画像ファイルの消去###########################################################
        #     #cleanFiles()


        # else:
        #     print("画像データ待機中")
        #     time.sleep(1)
    
    cv2.destroyAllWindows()
    print("complete")
"""
