from gui.box_config import BoxConfig
from kivy.garden.contextmenu import ContextMenuTextItem
import kivy.garden.contextmenu
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config
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
import os
import kivy
import serial

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
from communicate.slack import SlackBot
from move import Move
kivy.require('1.1.1')
current_path = os.path.dirname(os.path.abspath(__file__))
config_path = current_path + "/gui/config.ini"
Config.read(current_path)

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

        token = "xoxb-747399510036-750051690710-kGAhp4qvKZdynosrYrPEuGJd"  # 取得したトークン
        channel = "CMZRH5VR6"  # チャンネルID
        slack_bot = SlackBot(token, channel)
        weight_dist = 1
        weight_diff = 100
        self.data_path = data_path
        self.testdata_path = testdata_path
        self.box_list_path = box_list_path
        self.find_parts = find_parts
        self.tf2machine = tf2machine
        self.weight_dist = weight_dist
        self.weight_diff = weight_diff

    def say_hello(self, text):
        self.ids.select_number.id = text.id
        self.ids.com.color = 0, 1, 0, 1
        self.ids.com.text = "                                               COM:" + \
            str(text.id)

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

    image_texture = ObjectProperty(None)
    image_capture = ObjectProperty(None)

    def play(self):
        #global flgPlay
        #flgPlay = not flgPlay
        # if flgPlay == True:
        # self.image_capture = cv2.VideoCapture(0)
                # serial通信
        myserial = MySerial()
        myserial.init_port(self.ids.select_number.id)
        self.box_df = pd.read_csv(self.box_list_path, header=0)
        self.move = Move(self.data_path, self.testdata_path, self.find_parts,
                         self.box_df, self.tf2machine, myserial, self.weight_dist, self.weight_diff)
        print("play")
        Clock.schedule_interval(self.update, 1.0)
        # else:
        #    Clock.unschedule(self.update)
        #    self.image_capture.release()

    def stop(self):
        Clock.unschedule(self.update)

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
        err = self.move.run(0.7)
        if isinstance(err, list):
            pass
        else:
            if self.move.errs[err] == "no_parts":
                pass
                # slack_bot.send("finish parts sort",
                #                "finish", tempsave_path + "/ar.png")
                # cv2.waitKey(0)
                # cv2.waitKey(500)
                # カスケードファイルを指定して検出器を作成
                #cascade_file = "haarcascade_frontalface_alt.xml"
                #cascade = cv2.CascadeClassifier(cascade_file)
                # ここにopencvの処理入れて
                # str(self.manager.get_screen('test').port_number)がポート名

                # カメラ映像を上下左右反転2
        self.add_img(self.move.find_parts.ar_im, "camera_1")
        self.add_img(self.move.class_result_img, "camera_2")
        self.add_img(self.move.target_img, "camera_3")


class PartsSorterApp(App):
    def build(self):

        screen_manager.add_widget(StartMenu(name='start'))
        screen_manager.add_widget(BoxConfig(name='parts'))

        return screen_manager


if __name__ == '__main__':
    PartsSorterApp().run()
