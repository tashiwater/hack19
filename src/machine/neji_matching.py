#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd


class Obj():
    def __init__(self, x_m, y_m):
        self.x_m = x_m
        self.y_m = y_m


class NejiMatch():
    def __init__(self, box_df):
        self.box_df = box_df
        self.template_length = box_df.length

    def get_test_data(self, get_neji_output):
        self.objs = []
        self.raw_imgs = []
        self.diagonal = []
        if len(get_neji_output) < 1:
            return False
        for length, posi, img in get_neji_output:
            self.objs.append(Obj(posi[0], posi[1]))
            self.raw_imgs.append(img)
            self.diagonal.append(length)
        return True

    def predict(self):
        # print("self.diagonal", self.diagonal)
        self.df = pd.DataFrame(columns=["class_id", "score", "length"])
        # print("self.diagonal", self.diagonal)
        for length in self.diagonal:
            temp = abs(self.template_length - length * 1000)
            # temp = temp / self.template_length
            # print(temp)
            self.df = self.df.append(
                pd.Series([temp.idxmin(), temp.min(), length],
                          index=self.df.columns), ignore_index=True)
        self.df.sort_values(ascending=True, inplace=True, by="score")

    def paint(self, cont_list):
        for i in self.df.index:
            print(i)
