#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd


class NejiMatch():
    def __init__(self, box_df, weight_dist, weight_diff):
        self.box_df = box_df
        self.template_length = box_df.length
        print("self.template_length1", self.template_length )
        self.weight_dist = weight_dist
        self.weight_diff = weight_diff

    def get_test_data(self, get_neji_output):
        self.objs = []
        self.raw_imgs = []
        self.diagonal = []
        if len(get_neji_output) < 1:
            return False
        for length, posi, img in get_neji_output:
            self.objs.append([posi[0], posi[1]])
            self.raw_imgs.append(img)
            self.diagonal.append(length)
        self.objs = np.asarray(self.objs)
        return True

    def predict(self):
        # print("self.diagonal", self.diagonal)
        self.df = pd.DataFrame(
            columns=["class_id", "diff_to_template", "length", "dist", "score"])

        # print("self.diagonal", self.diagonal)
        for length in self.diagonal:
            diff_to_template = abs(self.template_length - length * 1000 + 3)
            self.df = self.df.append(
                pd.Series([diff_to_template.idxmin(), diff_to_template.min(), length, 10000, 10000],
                          index=self.df.columns), ignore_index=True)
        print("self.template_length2", self.template_length )
        for index in range(len(self.df)):
            posi = self.objs[index]
            for i, obj in enumerate(self.objs):
                if i == index:
                    continue
                dist = np.linalg.norm(posi - obj)
                # dist = 0
                if dist < self.df.dist[index]:
                    self.df.dist[index] = dist
            # self.df.score[index] = self.df.diff_to_template[index]
            self.df.score[index] = 1/self.df.dist[index] * self.weight_dist + \
                self.df.diff_to_template[index] * self.weight_diff
        self.df.sort_values(ascending=True, inplace=True, by="score")
