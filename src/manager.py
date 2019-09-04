#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import cv2
from cv2 import aruco
import signal

from AR.ar_detect import ARDetect
from AR.locate2d import Locate2d
from AR.find_parts import FindParts
from AR.transform import Transfrom2Machine
from communicate.serial_to_mbed import MySerial
from move import Move
# from machine.learn import Learn, plot_history

if __name__ == "__main__":
    # ctr-Cで消せるようにする
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # path
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../data"
    camera_info_path = data_path + "/camera_info"
    testdata_path = data_path + "/test"
    tempsave_path = data_path + "/temp"
    box_list_path = data_path + "/box_list.csv"
    checkpoint_path = data_path+"/checkpoint/cp.ckpt"
    train_path = data_path + "/train"

    # ARマーカー検出用クラス
    camera_matrix = np.loadtxt(camera_info_path + "/cameraMatrix.csv",
                               delimiter=",")
    distCoeffs = np.loadtxt(
        camera_info_path + "/distCoeffs.csv", delimiter=",")
    ar_marker_size = 0.02  # ARマーカー一辺[m]
    ar_detect = ARDetect(ar_marker_size, aruco.DICT_4X4_50,
                         camera_matrix, distCoeffs)

    # 対象領域決定用クラス
    x_scale_m = 0.214  # ARマーカの間隔（端から端）
    y_scale_m = 0.154
    remove_x_m = 0.020  # ROIと認識しない領域
    remove_y_m = 0.001
    locate_2d = Locate2d(ar_detect, x_scale_m, y_scale_m,
                         remove_x_m, remove_y_m)

    # カメラから画像を取得する関数
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    def get_frame():
        for i in range(5):
            _, frame = cap.read()
        # frame = cv2.imread(tempsave_path + "/raw.png")
        return frame
    # パーツ検出
    find_parts = FindParts(testdata_path, tempsave_path,
                           locate_2d, get_frame)

    # 座標変換用
    tf2machine = Transfrom2Machine(offset_x_mm=190, offset_y_mm=170)

    # serial通信
    myserial = MySerial()
    myserial.init_port()


# ボタンでモード選択
    mode = "move"
    box_df = pd.read_csv(box_list_path, header=0)
    if mode == "move":
        move = Move(data_path, testdata_path, find_parts,
                    box_df, tf2machine, myserial)
        while True:
            if move.run() is True:
                pass
            else:
                pass
            cv2.waitKey(500)
