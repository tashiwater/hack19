#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import rospy

import numpy as np
from cv2 import aruco
import os
import cv2
from AR.ar_detect import ARDetect
from AR.locate2d import Locate2d
from AR.find_parts import FindParts

if __name__ == "__main__":
    rospy.init_node("find_parts_node")
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../data"
    camera_info_path = data_path + "/camera_info"
    testdata_path = data_path + "/test"
    tempsave_path = data_path + "/temp"
    
    camera_matrix = np.loadtxt(camera_info_path + "/cameraMatrix.csv",
        delimiter = ",")
    distCoeffs = np.loadtxt(
            camera_info_path + "/distCoeffs.csv", delimiter=",")
    ar_marker_size=0.02  # ARマーカー一辺[m]
    ar_detect = ARDetect(ar_marker_size, aruco.DICT_4X4_50,
    camera_matrix, distCoeffs)

    locate_2d = Locate2d(ar_detect, 0.2, 0.159, 0.017, 0.005)
    # cap=cv2.VideoCapture(0) 
    # cap.set(3, 320)
    # cap.set(4, 240)
    # cap.set(5, 5)


    def get_frame():
        # _, frame = cap.read()
        frame = cv2.imread(data_path + "/temp.jpg")
        return frame

    find_parts_srv=rospy.get_param("~find_parts_srv", "find_parts_srv")
    
    find_parts = FindParts( camera_info_path, testdata_path, tempsave_path,
                 locate_2d, get_frame, find_parts_srv)
    find_parts.get_testdata()
    #trainデータ作成モード
    traindata_path = data_path + "/train"
    while not rospy.is_shutdown():
        find_parts.make_traindata(traindata_path)
    
    rospy.spin()
