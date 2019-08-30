#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
from cv2 import aruco
import pandas as pd
import os
import cv2
from AR.ar_detect import ARDetect
from AR.locate2d import Locate2d
from AR.find_parts import FindParts
from AR.transform import Transfrom2Machine
from machine.matching import Match

import signal

if __name__ == "__main__":
    # ctr-Cで消せるようにする
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../data"
    camera_info_path = data_path + "/camera_info"
    testdata_path = data_path + "/test"
    tempsave_path = data_path + "/temp"
    checkpoint_path = data_path+"/checkpoint/cp.ckpt"
    
    camera_matrix = np.loadtxt(camera_info_path + "/cameraMatrix.csv",
                               delimiter=",")
    distCoeffs = np.loadtxt(
        camera_info_path + "/distCoeffs.csv", delimiter=",")
    ar_marker_size = 0.02  # ARマーカー一辺[m]
    ar_detect = ARDetect(ar_marker_size, aruco.DICT_4X4_50,
                         camera_matrix, distCoeffs)
    locate_2d = Locate2d(ar_detect, 0.2, 0.159, 0.017, 0.005)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    def get_frame():
        for i in range(5):
            _, frame = cap.read()
        # frame = cv2.imread(data_path + "/temp.jpg")
        return frame

    find_parts = FindParts(camera_info_path, testdata_path, tempsave_path,
                           locate_2d, get_frame)

    match = Match(data_path, testdata_path)

    box_list_path = data_path + "/box_list.csv"
    box_df = pd.read_csv(box_list_path, header=0)
    tf2machine = Transfrom2Machine(194, 198)
    while True:
        cv2.waitKey(1000)
        if find_parts.get_testdata() is False:
            cv2.waitKey(500)
            continue
        cv2.waitKey(500)
        if match.get_test_data() is False:
            continue
        match.predict()
        use_obj = match.df.index[0]
        print("target_img", use_obj)
        x = match.objs[use_obj].x_m
        y = match.objs[use_obj].y_m
        z = match.max_index[use_obj]
        print("img posi", x, y)
        print("class", z)

        to_mbed_x, to_mbed_y = tf2machine.get_xy_mm(
            match.objs[use_obj].x_m, match.objs[use_obj].y_m)
        id = int(z)
        box = box_df.iloc[id]
        to_mbed_z = box.box_x
        to_mbed_w = box.box_y
        print("pub to mbed ", to_mbed_x, to_mbed_y, to_mbed_z, to_mbed_w)

        cv2.waitKey(0)
