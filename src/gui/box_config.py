#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
import pandas as pd
import os


class BoxConfig(Screen, BoxLayout):
    def __init__(self, **kwargs):
        super(BoxConfig, self).__init__(**kwargs)
        current_path = os.path.dirname(os.path.abspath(__file__))
        data_path = current_path + "/../../data"
        self.neji_template_img_path = data_path + "/gui/neji"
        self.box_list_path = data_path + "/box_list.csv"
        box_df = pd.read_csv(self.box_list_path, header=0)
        self.ids.text_a.text = box_df.name[0]
        self.ids.text_b.text = box_df.name[1]
        self.ids.text_c.text = box_df.name[2]
        self.ids.text_d.text = box_df.name[3]
        self.ids.slider_a.value = int(box_df.length[0])
        self.ids.slider_b.value = int(box_df.length[1])
        self.ids.slider_c.value = int(box_df.length[2])
        self.ids.slider_d.value = int(box_df.length[3])

    def save(self):
        box_df = pd.read_csv(self.box_list_path, header=0)
        box_df.loc[0, "name"] = self.ids.text_a.text
        box_df.loc[1, "name"] = self.ids.text_b.text
        box_df.loc[2, "name"] = self.ids.text_c.text
        box_df.loc[3, "name"] = self.ids.text_d.text

        box_df.loc[0, "length"] = self.ids.slider_a.value
        box_df.loc[1, "length"] = self.ids.slider_b.value
        box_df.loc[2, "length"] = self.ids.slider_c.value
        box_df.loc[3, "length"] = self.ids.slider_d.value
        box_df.to_csv(self.box_list_path, index=False)

    # 以下不要
    def get_text_a(self, *args):
        print("A_parts")
        print("name: " + str(self.ids.text_a.text))
        print("length: " + str(self.ids.slider_a.value))

    def get_text_b(self, *args):
        print("B_parts")
        print("name: " + str(self.ids.text_b.text))
        print("length: " + str(self.ids.slider_b.value))

    def get_text_c(self, *args):
        print("C_parts")
        print("name: " + str(self.ids.text_c.text))
        print("length: " + str(self.ids.slider_c.value))

    def get_text_d(self, *args):
        print("D_parts")
        print("name: " + str(self.ids.text_d.text))
        print("length: " + str(self.ids.slider_d.value))
    
    def change_image(self, id, path):
        change_id = self.ids[id]
        print(self.neji_template_img_path + path + ".jpg")
        change_id.source = self.neji_template_img_path + "/" + path + ".jpg"
        print(path)
        print(id)
        print(change_id.source)
