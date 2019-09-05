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
from gui.box_config import BoxConfig
kivy.require('1.1.1')


current_path = os.path.dirname(os.path.abspath(__file__))
config_path = current_path + "/gui/config.ini"
Config.read(current_path)


resource_add_path("C:\Windows\Fonts")
LabelBase.register(DEFAULT_FONT, "UDDigiKyokashoN-R.ttc")

screen_manager = ScreenManager()


class StartMenu(Screen):
    ''' スタートメニュー '''

    def say_hello(self, text):
        self.ids.select_number.id = text.id
        self.ids.com.color = 0, 1, 0, 1
        self.ids.com.text = "                                           COM:" + text.id

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

            image_texture = ObjectProperty(None)
    image_capture = ObjectProperty(None)

    def play(self):
        #global flgPlay
        #flgPlay = not flgPlay
        # if flgPlay == True:
        self.image_capture = cv2.VideoCapture(0)
        print("play")
        Clock.schedule_interval(self.update, 1.0 / 20)
        # else:
        #    Clock.unschedule(self.update)
        #    self.image_capture.release()

    def stop(self):
        Clock.unschedule(self.update)

    def update(self, dt):
        ret, frame = self.image_capture.read()
        if ret:
            # カスケードファイルを指定して検出器を作成
            #cascade_file = "haarcascade_frontalface_alt.xml"
            #cascade = cv2.CascadeClassifier(cascade_file)
            # ここにopencvの処理入れて
            # str(self.manager.get_screen('test').port_number)がポート名

            # カメラ映像を上下左右反転
            buf = cv2.flip(frame, -1)
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(
                buf.tostring(), colorfmt='bgr', bufferfmt='ubyte')
            camera = self.ids['camera']
            camera.texture = image_texture


class PartsSorterApp(App):
    def build(self):

        screen_manager.add_widget(StartMenu(name='start'))
        screen_manager.add_widget(BoxConfig(name='parts'))

        return screen_manager


if __name__ == '__main__':
    PartsSorterApp().run()
