#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#ARからROIを抽出するクラス
import numpy as np
class Locate2d():
    def __init__(self, ar_detect, x_scale_m, y_scale_m, remove_x_m, remove_y_m):
        self.ar_detect = ar_detect
        self.x_scale_m = x_scale_m
        self.y_scale_m = y_scale_m
        self.remove_x_m = remove_x_m
        self.remove_y_m = remove_y_m
    # ROIの画像を切り取る
    def get_roi_img(self):
        x_min = []
        x_max = []
        y_min = []
        y_max = []

        corners = [self.ar_detect.get_corner(i) for i in range(4)]
        print(corners)
        if len(corners[0]) > 1:
            x_min.append(corners[0][1][0])
            y_min.append(corners[0][1][1])
        if len(corners[1]) > 1:
            x_min.append(corners[1][2][0])
            y_max.append(corners[1][2][1])
        if len(corners[2]) > 1:
            x_max.append(corners[2][3][0])
            y_max.append(corners[2][3][1])
        if len(corners[3]) > 1:
            x_max.append(corners[3][0][0])
            y_min.append(corners[3][0][1])

        print("x_min", x_max, y_min, y_max)
        #どれか一つでも取れなければreturn
        if len(x_min) < 1 or len(x_max) < 1 or len(y_min) < 1 or len(y_max) < 1:
            return None
        #候補の平均をとる
        corner = [[np.mean(x_min), np.mean(y_min)],
                  [np.mean(x_max), np.mean(y_max)]]
        corner = np.vectorize(int)(corner)
        self.roi_img = self.ar_detect.img[corner[0][1]:corner[1][1], corner[0][0]:corner[1][0]]
        self.corner = corner
        self.x_scale_px = abs(corner[1][0] - corner[0][0])
        self.y_scale_px = abs(corner[1][1] - corner[0][1])


        remove_x =int(self.remove_x_m / self.x_scale_m * self.x_scale_px)
        remove_y =int(self.remove_y_m / self.y_scale_m * self.y_scale_px)
        self.roi_img = self.roi_img[remove_y:-remove_y, remove_x:-remove_x]
        self.roi_x_scale_m = self.x_scale_m - 2 * self.remove_x_m
        self.roi_y_scale_m = self.y_scale_m - 2 * self.remove_y_m
        return self.roi_img

    def get_z_pred(self):
        z_s = []
        for i in range(4):
            rvec, posi = self.ar_detect.get_ar_posi(i)
            if len(posi) < 1:
                continue
            z_s.append(posi[2])
        return np.mean(z_s)

    #roi内での画素をmで返す
    def pred_posi_in_roi(self, at):
        height, width = self.roi_img.shape[:2]
        x_m = at[0] / width * self.roi_x_scale_m
        y_m = at[1] / height * self.roi_y_scale_m
        return x_m, y_m

