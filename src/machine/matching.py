#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os

# import rospy
# from geometry_msgs.msg import Vector3
# from std_srvs.srv import SetBool, SetBoolResponse

from machine.Object import RecognizedObject
from machine.img_generator import padding
from machine.learn import Learn


class Match():
    def __init__(self, data_path, testdata_path, box_df):
        self.testdata_path = testdata_path
        learn = Learn(data_path, box_df)
        self.img_size = learn.img_size
        self.model = learn.create_model()
        self.model.load_weights(learn.checkpoint_path)
        # while not rospy.is_shutdown() and self.match() is False:
        #     pass

    def match(self):
        print("matching")
        if self.get_test_data() is False:
            return False
        self.predict()
        return True

    def get_test_data(self):
        self.objs = []
        images = []
        self.raw_imgs = []
        xml_dirs = os.listdir(self.testdata_path+"/xmls")
        print(xml_dirs)
        if len(xml_dirs) < 1:
            print("there is no xml")
            return False
        # オブジェクトデータの取得
        for f in xml_dirs:
            obj = RecognizedObject(self.testdata_path+'/xmls/' + f)
            # img = padding(obj.image)
            # print(obj.image.size)
            # cv2.imshow("img", obj.image)
            # cv2.waitKey(0)
            self.raw_imgs.append(obj.image)
            img = cv2.resize(obj.image, self.img_size)
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

# ROS関連
    def set_ros(self, pub_topic, match_srv, get_target_srv):
        self.pub = rospy.Publisher(pub_topic, Vector3, queue_size=1)
        rospy.Service(match_srv, SetBool, self.srv_callback)
        rospy.Service(get_target_srv, SetBool, self.get_target_srv_callback)

    def srv_callback(self, request):
        resp = SetBoolResponse()
        resp.message = "called. data: " + str(request.data)
        print(resp.message)
        resp.success = self.match()
        return resp

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
