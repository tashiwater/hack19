#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import serial.tools.list_ports
import serial
import pandas as pd
from cv2 import aruco
import signal
from decimal import Decimal, ROUND_HALF_UP
import urllib.request, urllib.error
import webbrowser

from AR.ar_detect import ARDetect
from AR.locate2d import Locate2d
from AR.find_parts import FindParts
from AR.transform import Transfrom2Machine
from communicate.serial_to_mbed import MySerial
from communicate.slack import SlackBot
from machine.move import Move
from json_scp.gspread_change import SpreadManager
from gui.box_config import BoxConfig
import os


current_path = os.path.dirname(os.path.abspath(__file__))

# from machine.learn import Learn, plot_history
class StartMenu():
    ''' スタートメニュー '''

    def __init__(self, **kwargs):
        # ctr-Cで消せるようにする
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        # path
        data_path = current_path + "/../data"
        camera_info_path = data_path + "/camera_info"
        testdata_path = data_path + "/test"
        tempsave_path = data_path + "/temp"
        box_list_path = data_path + "/box_list.csv"
        self.param_path = data_path + "/params.csv"
        self.gui_path = data_path + "/gui"
        self.sozai_path = self.gui_path + "/sozai"
        self.spread_url = data_path + '/PartsList-8533077dcf0f.json'
        self.spread_name = 'PartsList'
        #通信関係ONOFF
        self.serial_on = False
        self.spreadManager = None

        self.param_df = pd.read_csv(self.param_path, header=None, index_col=0)
        print(self.param_df)
        print("type", type(self.param_df))


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

        token = "xoxb-747399510036-750051690710-f4DUOcItFO9geAZySmdCsZm4"  # 取得したトークン
        channel = "CMZRH5VR6"  # チャンネルID
        
        self.slack_bot = SlackBot(token, channel)
        self.data_path = data_path
        self.testdata_path = testdata_path
        self.box_list_path = box_list_path
        self.find_parts = find_parts
        self.tf2machine = tf2machine
        self.weight_dist = 1
        self.weight_diff = 100
        self.myserial = MySerial()
        self.wifi_ok = False
        self.slack_on = False
        self.googlesheet_on = False
        self.line_on = False
        self.is_playing = False
        self.wifi_icon_path = [self.sozai_path + "/Network_ON_Color.png",
                                self.sozai_path + "/Network_OFF_.png"]
        self.slack_icon_path = [self.sozai_path + "/Slack_ON.png"
                                ,self.sozai_path + "/Slack_OFF_.png"]
        self.googlesheet_icon_path = [self.sozai_path + "/GoogleSheet_ON_Color.png",
                                self.sozai_path + "/GoogleSheet_OFF_.png"]
        self.line_icon_path = [self.sozai_path + "/LINE_ON_Color.png",
                                self.sozai_path + "/LINE_OFF_.png"]

    def play(self):
        print("play")
        if self.serial_on:
            if self.myserial.init_port(self.myserial.search_com_port()) is False:
                return
            self.myserial.mbed_reset()
        
        self.box_df = pd.read_csv(self.box_list_path, header=0)

        if self.googlesheet_on:
            self.spreadManager = SpreadManager(self.spread_name, self.spread_url)
        else:
            self.spreadManager = None
        self.move = Move(self.data_path, self.testdata_path, self.find_parts,
                         self.box_df, self.tf2machine, self.myserial, self.weight_dist, self.weight_diff,self.spreadManager)
        
    def update(self, dt):
        
        if self.serial_on:
            err = self.move.run(self.param_df.loc["solenoid_duty_small",1], 
            self.param_df.loc["solenoid_duty_big",1],
            self.param_df.loc["gray_threshold",1])
        else:
            err = self.move.no_serial_run(self.param_df.loc["solenoid_duty_big",1], 
            self.param_df.loc["solenoid_duty_small",1],
            self.param_df.loc["gray_threshold",1])
        if isinstance(err, list):
            pass
        else:
            print("move ret", self.move.errs[err])
            if self.move.errs[err] == "no_parts":
                # pass
                print("err in ")
                if self.slack_on:
                    print("slack")
                    # self.slack_bot.send("finish parts sort",
                    #                     "finish", self.find_parts.tempsave_path + "/ar.png")
                print("finish")
                self.stop()
    
    def threshold_test(self):
        self.spreadManager = None
        self.box_df = pd.read_csv(self.box_list_path, header=0)
        self.move = Move(self.data_path, self.testdata_path, self.find_parts,
                         self.box_df, self.tf2machine, self.myserial, self.weight_dist, self.weight_diff,self.spreadManager)
        err = self.move.no_serial_run(self.param_df.loc["solenoid_duty_big",1], 
            self.param_df.loc["solenoid_duty_small",1],
            self.param_df.loc["gray_threshold",1])
        if isinstance(err, list):
            self.add_img(self.move.find_parts.ar_im, "ar_img")
            self.add_img(self.move.class_result_img, "roi_img")

if __name__ == "__main__":
    # ctr-Cで消せるようにする
    start = StartMenu()
    start.play()
    start.update(5)