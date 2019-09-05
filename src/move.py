#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import cv2
# from machine.matching import Match
from machine.neji_matching import NejiMatch
import matplotlib.pyplot as plt
import numpy as np


class Move():
    def __init__(self, data_path, testdata_path, find_parts, box_df, tf2machine, myserial, weight_dist, weight_diff):
        self.find_parts = find_parts
        self.box_df = box_df

        # self.match = Match(data_path, testdata_path, box_df)
        self.match = NejiMatch(box_df, weight_dist, weight_diff)
        self.tf2machine = tf2machine
        self.myserial = myserial
        cm_name = 'jet'
        self.color_map = plt.get_cmap(cm_name, len(self.box_df))
        self.errs = ["nothing", "no_parts", "something"]
        self.class_result_img = None
        self.target_img = None

    def no_serial_run(self):
        if self.find_parts.get_testdata() is False:
            return self.errs.index("something")
        cv2.waitKey(500)
        if self.match.get_test_data(self.find_parts.get_neji_output()) is False:
            return self.errs.index("no_parts")
        self.match.predict()
        self.class_result_img = self.paint(self.find_parts.cont.size_list,
                                           self.match, self.find_parts.roi_img)

        self.FindStock()

        print(self.match.df)
        use_obj = self.match.df.index[0]
        x = self.match.objs[use_obj][0]
        y = self.match.objs[use_obj][1]
        z = self.match.df.head(1)["class_id"]
        self.target_img = self.match.raw_imgs[use_obj]
        # cv2.imshow("target", self.match.raw_imgs[use_obj])

        print("img posi", x, y)
        print("target", self.match.df.head(1))

        to_mbed_x, to_mbed_y = self.tf2machine.get_xy_mm(x, y)

        pick_place = [100, 20]
        if self.match.df.is_stock[use_obj] == 0:
            id = int(z)
            box = self.box_df.iloc[id]
            to_mbed_z = box.box_x
            to_mbed_w = box.box_y
            solenoid = 0.5
        else:
            to_mbed_z = pick_place[0]
            to_mbed_w = pick_place[1]
            solenoid = 0.3

        print("pub to mbed ", to_mbed_x, to_mbed_y,
              to_mbed_z, to_mbed_w, solenoid)
        double_list = [to_mbed_x, to_mbed_y, to_mbed_z, to_mbed_w]
        pub_list = list(map(int, double_list))
        pub_list.append(solenoid)
        return pub_list

    def run(self, solenoid):
        self.back_str = self.myserial.buffer_read(20)
        if "go" not in self.back_str:
            return self.errs.index("something")
        while "wait_target" not in self.myserial.buffer_read(20):
            pass
        # while True:
        int_list = self.no_serial_run()
        if not isinstance(int_list, list):
            return int_list
        self.myserial.write(int_list)
        self.back_str = self.myserial.read(30)
        return self.errs.index("nothing")

    def FindStock(self):
        separate_x_m = 0.1
        self.match.df["is_stock"] = 0
        for i in range(len(self.match.df)):
            if self.match.objs[i][0] < separate_x_m:
                self.match.df["is_stock"][i] = 1
        self.match.df.sort_values(
            ascending=True, inplace=True, by=["is_stock", "score"])

    def paint(self, cont_list, match, roi_img):
        show = roi_img.copy()
        for index, row in match.df.iterrows():

            clas = int(row["class_id"])
            # print("class", clas)
            # bgra = np.asarray(self.color_map(clas)) * 255
            # print("bgra", bgra)
            bgr = np.asarray(
                [match.box_df.color_b[clas], match.box_df.color_g[clas], match.box_df.color_r[clas]])
            cv2.rectangle(show, (cont_list[index][0], cont_list[index][2]), (cont_list[index][1], cont_list[index][3]),
                          bgr.tolist(), thickness=10)
        # self.find_parts.show_img("class", show)
        return show
