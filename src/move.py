#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import cv2
# from machine.matching import Match
from machine.neji_matching import NejiMatch
import matplotlib.pyplot as plt
import numpy as np


class Move():
    def __init__(self, data_path, testdata_path, find_parts, box_df, tf2machine, myserial):
        self.find_parts = find_parts
        self.box_df = box_df

        # self.match = Match(data_path, testdata_path, box_df)
        self.match = NejiMatch(box_df)
        self.tf2machine = tf2machine
        self.myserial = myserial
        cm_name = 'jet'
        self.color_map = plt.get_cmap(cm_name, len(self.box_df))
        # for i in range(len(self.box_df)):
        #     print("color", self.color_map(i))

    def run(self):
        self.back_str = self.myserial.buffer_read(20)
        if "go" not in self.back_str:
            return False

        if self.find_parts.get_testdata() is False:
            return False
        cv2.waitKey(500)
        if self.match.get_test_data(self.find_parts.get_neji_output()) is False:
            return False
        self.match.predict()
        self.paint(self.find_parts.cont.size_list,
                   self.match.df, self.find_parts.roi_img)
        print(self.match.df)
        use_obj = self.match.df.index[0]
        x = self.match.objs[use_obj].x_m
        y = self.match.objs[use_obj].y_m
        z = self.match.df.head(1)["class_id"]
        cv2.imshow("target", self.match.raw_imgs[use_obj])

        print("img posi", x, y)
        print("target", self.match.df.head(1))

        to_mbed_x, to_mbed_y = self.tf2machine.get_xy_mm(x, y)
        id = int(z)
        box = self.box_df.iloc[id]
        to_mbed_z = box.box_x
        to_mbed_w = box.box_y
        print("pub to mbed ", to_mbed_x, to_mbed_y, to_mbed_z, to_mbed_w)
        double_list = [to_mbed_x, to_mbed_y, to_mbed_z, to_mbed_w]
        int_list = list(map(int, double_list))
        # self.back_str = self.myserial.buffer_read(20)
        while "wait_target" not in self.myserial.buffer_read(20):
            pass
        # while True:
        self.myserial.write(int_list)
        self.back_str = self.myserial.read(20)
        return True

    def paint(self, cont_list, df, roi_img):
        show = roi_img.copy()
        for index, row in df.iterrows():

            clas = int(row["class_id"])
            # print("class", clas)
            bgra = np.asarray(self.color_map(clas)) * 255
            # print("bgra", bgra)
            cv2.rectangle(show, (cont_list[index][0], cont_list[index][2]), (cont_list[index][1], cont_list[index][3]),
                          bgra[:3], thickness=10)
        self.find_parts.show_img("class", show)
