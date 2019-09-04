
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
from kivy.config import Config
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, StringProperty, OptionProperty
from kivy.uix.widget import Widget
from kivy.app import App

import os
import sys

import serial
import serial.tools.list_ports
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from box_config import BoxConfig

current_path = os.path.dirname(os.path.abspath(__file__))
config_path = current_path + "/../gui/config.ini"
class_path = current_path + "/../src/"
sys.path.append(class_path)

kivy.require('1.1.1')
Config.read(config_path)
resource_add_path("C:\Windows\Fonts")
LabelBase.register(DEFAULT_FONT, "UDDigiKyokashoN-R.ttc")

screen_manager = ScreenManager()


class StartMenu(Screen):
    ''' スタートメニュー '''
    pass


class SettingScreen(Screen):
    '''学習画面'''
    pass


class ActionScreen(Screen):
    '''動作画面'''
    image_texture = ObjectProperty(None)
    image_capture = ObjectProperty(None)

    def play(self):
        #global flgPlay
        #flgPlay = not flgPlay
        # if flgPlay == True:
        # self.image_capture = cv2.VideoCapture(0)

        print("play")
        Clock.schedule_interval(self.update, 1.0 / 20)
        # else:
        #    Clock.unschedule(self.update)
        #    self.image_capture.release()

    def update(self, dt):
        ret, frame = self.image_capture.read()
        if ret:
            # カスケードファイルを指定して検出器を作成
            #cascade_file = "haarcascade_frontalface_alt.xml"
            #cascade = cv2.CascadeClassifier(cascade_file)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2値化
            retval, bw = cv2.threshold(
                gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # 輪郭を抽出
            contours, hierarchy = cv2.findContours(
                bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # 矩形検出された数（デフォルトで0を指定）
            detect_count = 0
            # 各輪郭に対する処理
            for i in range(0, len(contours)):
                # 輪郭の領域を計算
                area = cv2.contourArea(contours[i])
                # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
                if area < 1e2 or 1e5 < area:
                    continue
                # 外接矩形
                if len(contours[i]) > 0:
                    rect = contours[i]
                    x, y, w, h = cv2.boundingRect(rect)
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    detect_count = detect_count + 1

            # カメラ映像を上下左右反転
            buf = cv2.flip(frame, -1)
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(
                buf.tostring(), colorfmt='bgr', bufferfmt='ubyte')
            camera = self.ids['camera']
            camera.texture = image_texture

    def detect_contour(path):
        # 画像を読込
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        # グレースケール画像へ変換
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # 2値化
        retval, bw = cv2.threshold(
            gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 輪郭を抽出
        contours, hierarchy = cv2.findContours(
            bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # 矩形検出された数（デフォルトで0を指定）
        detect_count = 0
        # 各輪郭に対する処理
        for i in range(0, len(contours)):
            # 輪郭の領域を計算
            area = cv2.contourArea(contours[i])
            # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
            if area < 1e2 or 1e5 < area:
                continue
            # 外接矩形
            if len(contours[i]) > 0:
                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)
                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detect_count = detect_count + 1
        # 外接矩形された画像を表示
        #cv2.imshow('output', src)

    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class testInput(Screen, BoxLayout):
    def test_a(self):
        slider_d = self.manager.get_screen('text').slider_d.value
        print("test_a accessed")
        print("length: " + str(slider_d))

    def test_b(self):
        pass
        #print("test_b accessed")

    def say_hello(self, text):
        self.ids['context_menu'].hide()
        self.ids['sel_btn'].center_x = self.center_x*3.7 - self.width
        self.ids.select_box.id = text.id
        self.ids.select_box.text = text.text

    def add_text(self, text):
        self.hello = ContextMenuTextItem(text=str(text), id=str(text.device))
        self.hello.bind(on_press=self.say_hello)
        self.ids['context_menu'].add_widget(self.hello)

    def search_com_port(self):
        # self.ids['context_menu'].clear_widgets()
        coms = serial.tools.list_ports.comports()
        if coms == []:
            print("Connection Failed")
        else:
            for com in coms:
                self.add_text(com)

    def select_port(self):
        use_port = self.ids.select_box.id
        print('Use COM port: ' + use_port)
        ser = serial.Serial(use_port, 9600, timeout=3)
        # ser.flushInput()
        print('port_open')
        r_data = ser.read_until(size=30)  # size分Read
        got_str = r_data.decode(encoding="utf-8")
        print('Recv1: ' + got_str)
        r_data = ser.read_until(size=30)  # size分Read
        got_str = r_data.decode(encoding="utf-8")
        print('Recv2: ' + got_str)

        data_str = str("fuck")
        ser.write(data_str.encode(encoding='utf-8'))
        ser.close()

        # r_data = ser.read_until(size=30)  # size分Read
        # got_str = r_data.decode(encoding="utf-8")
        # print('Recv3: ' + got_str)
        # r_data = ser.read_until(size=30)  # size分Read
        # got_str = r_data.decode(encoding="utf-8")
        # print('Recv4: ' + got_str)

        # ser.close()
        # return use_port


class TestApp(App):
    def build(self):

        screen_manager.add_widget(StartMenu(name='start'))
        screen_manager.add_widget(SettingScreen(name='set'))
        screen_manager.add_widget(ActionScreen(name='act'))
        screen_manager.add_widget(BoxConfig(name='text'))
        screen_manager.add_widget(testInput(name='test'))

        return screen_manager

    def reset_game(self):
        print("reset")


if __name__ == '__main__':
    TestApp().run()
