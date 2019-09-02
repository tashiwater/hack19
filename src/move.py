#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import cv2
# from machine.matching import Match
from machine.neji_matching import NejiMatch


class Move():
    def __init__(self, data_path, testdata_path, find_parts, box_df, tf2machine, myserial):
        self.find_parts = find_parts
        self.box_df = box_df
        # self.match = Match(data_path, testdata_path, box_df)
        self.match = NejiMatch(box_df)
        self.tf2machine = tf2machine
        self.myserial = myserial

    def run(self):
        if self.find_parts.get_testdata() is False:
            return False
        cv2.waitKey(500)
        if self.match.get_test_data(self.find_parts.get_neji_output()) is False:
            return False
        self.match.predict()
        self.match.paint(self.find_parts.cont.size_list)
        print(self.match.df)
        use_obj = self.match.df.index[0]
        x = self.match.objs[use_obj].x_m
        y = self.match.objs[use_obj].y_m
        z = self.match.df.head(1)["class_id"]
        cv2.imshow("target_img_class" + str(z), self.match.raw_imgs[use_obj])

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
        self.myserial.write(int_list)
        self.myserial.read(100)  # callbackを確認
        return True
