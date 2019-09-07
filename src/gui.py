#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# import os
from kivy.config import Config
import os


current_path = os.path.dirname(os.path.abspath(__file__))
config_path = current_path + "/gui/config.ini"
Config.read(config_path)

import kivy
from kivy.garden.contextmenu import ContextMenuTextItem
import kivy.garden.contextmenu
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, StringProperty, OptionProperty
from kivy.uix.widget import Widget
from kivy.app import App

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

Builder.load_file(current_path + '/gui/StartMenu.kv')
Builder.load_file(current_path + '/gui/BoxConfig.kv')
kivy.require('1.1.1')

resource_add_path("C:\Windows\Fonts")

LabelBase.register(DEFAULT_FONT, "UDDigiKyokashoN-R.ttc")

screen_manager = ScreenManager()


class StartMenu(Screen):
    ''' スタートメニュー '''

    def __init__(self, **kwargs):
        super(StartMenu, self).__init__(**kwargs)
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
        self.serial_on = True
        self.spreadManager = None

        self.param_df = pd.read_csv(self.param_path, header=None, index_col=0)
        self.ids.slider_speed.value = float(self.param_df.loc["solenoid_duty_small",1])
        self.ids.slider_pwm.value = float(self.param_df.loc["solenoid_duty_big",1])
        self.ids.slider_gray.value = int(self.param_df.loc["gray_threshold",1])
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
        Clock.schedule_interval(self.check_Wi_Fi, 10)
        self.is_playing = False
        self.wifi_icon_path = [self.sozai_path + "/Network_ON_Color.png",
                                self.sozai_path + "/Network_OFF_.png"]
        self.slack_icon_path = [self.sozai_path + "/Slack_ON.png"
                                ,self.sozai_path + "/Slack_OFF_.png"]
        self.googlesheet_icon_path = [self.sozai_path + "/GoogleSheet_ON_Color.png",
                                self.sozai_path + "/GoogleSheet_OFF_.png"]
        self.line_icon_path = [self.sozai_path + "/LINE_ON_Color.png",
                                self.sozai_path + "/LINE_OFF_.png"]

    def emergency_button_on(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play()
        else:
            self.stop()
    
    


    def check_Wi_Fi(self, dt):
        # try:
        #     urllib.request.urlopen(url='https://www.google.com/', timeout=1)
        # except Exception as e:
        #     print(e)
        #     self.wifi_ok = 0
        # else:
        #     self.wifi_ok = 1
        if self.wifi_ok:
            print("wifi connect")
            self.ids.wifi_icon.source =  self.wifi_icon_path[0]
        else:
            print("wifi false")
            self.ids.wifi_icon.source =  self.wifi_icon_path[1]
            self.ids.slack_icon.source = self.slack_icon_path[1]
            self.ids.line_icon.source = self.line_icon_path[1]
            self.ids.googlesheet_icon.source = self.googlesheet_icon_path[1]
        
    
    def say_hello(self, text):
        self.ids.select_number.id = text.id
        self.ids.port_name.text = "COM:" + str(text.id)

    def add_text(self, com):
        self.add = ContextMenuTextItem(text=str(com), id=str(com.device))
        self.add.bind(on_press=self.say_hello)
        self.ids['com_port'].add_widget(self.add)

    def search_com_port(self):
        self.ids['com_port'].clear_widgets()
        coms = serial.tools.list_ports.comports()
        if coms == []:
            print("Connection Failed")
            self.add = ContextMenuTextItem(text="none")
            self.add.bind(on_press=self.say_hello)
            self.ids['com_port'].add_widget(self.add)
        else:
            for com in coms:
                self.add_text(com)

    def nomal_mode(self):
        self.ids.slider_mode.pos_hint = {"center_x": 0.1, "center_y": 2}

    def extra_mode(self):
        self.ids.slider_mode.pos_hint = {"center_x": 0.5, "center_y": 0.6}

    def param_save(self):
        self.param_df.loc["solenoid_duty_small",1] = self.ids.slider_speed.value
        self.param_df.loc["solenoid_duty_big",1] = self.ids.slider_pwm.value
        self.param_df.loc["gray_threshold",1] = self.ids.slider_gray.value
        
        self.param_df.to_csv(self.param_path, header=False)
    def stop(self):
        self.is_playing = False
        Clock.unschedule(self.update)
        print("stop")

    def show_spreadsheet(self):
        if self.wifi_ok is False:
            return
        webbrowser.open(url="https://docs.google.com/spreadsheets/d/1wosAqe5n1EqrKXIEBkkgfs7rNXQ8HIOxru3Pt8eQRzs/edit#gid=0")

    def slack_button_down(self):
        if self.wifi_ok is False:
            return
        self.slack_on = not self.slack_on
        if self.slack_on:
            print("slack on")
            self.ids.slack_icon = self.slack_icon_path[0]
        else:
            self.ids.slack_icon = self.slack_icon_path[1]
    
    def googlesheet_down(self):
        if self.wifi_ok is False:
            return
        self.googlesheet_on = not self.googlesheet_on
        if self.googlesheet_on:
            self.show_spreadsheet()
            self.ids.googlesheet_icon = self.googlesheet_icon_path[0]
        else:
            self.ids.googlesheet_icon = self.googlesheet_icon_path[1]

    image_texture = ObjectProperty(None)
    image_capture = ObjectProperty(None)

    def play(self):
        print("play")
        if self.serial_on:
            if self.myserial.init_port(self.ids.select_number.id) is False:
                return
            self.myserial.mbed_reset()
        
        self.box_df = pd.read_csv(self.box_list_path, header=0)

        if self.googlesheet_on:
            self.spreadManager = SpreadManager(self.spread_name, self.spread_url)
        else:
            self.spreadManager = None
        self.move = Move(self.data_path, self.testdata_path, self.find_parts,
                         self.box_df, self.tf2machine, self.myserial, self.weight_dist, self.weight_diff,self.spreadManager)
        
        Clock.schedule_interval(self.update, 5.0)
        #    self.image_capture.release()

    def add_img(self, frame, camera_name):
        if frame is None:
            return
        buf = cv2.flip(frame, -1)
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(
            buf.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        camera_1 = self.ids[camera_name]
        camera_1.pos_hint = {"center_x": 0.6}
        camera_1.texture = image_texture

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
        self.add_img(self.move.find_parts.ar_im, "ar_img")
        self.add_img(self.move.class_result_img, "roi_img")
        self.add_img(self.move.target_img, "target_img")
    
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

class PartsSorterApp(App):
    def build(self):

        screen_manager.add_widget(StartMenu(name='start'))
        screen_manager.add_widget(BoxConfig(name='parts'))

        return screen_manager


if __name__ == '__main__':
    PartsSorterApp().run()
