#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Transfrom2Machine():
    def __init__(self, offset_x_mm, offset_y_mm):
        """
        @param
        offset_x_mm, _y: machine座標系で見た画像座標系の原点の座標
        """
        self.offset_x_mm = offset_x_mm
        self.offset_y_mm = offset_y_mm
    def get_xy_mm(self, x_m, y_m):
        #machine座標系と同じ方向になるよう回転
        ret_x = -float(y_m)
        ret_y = -float(x_m)
        # m → mmに変換
        ret_x *= 1000
        ret_y *= 1000

        #machine座標系で見た画像座標系の原点の座標を加える
        ret_x += self.offset_x_mm
        ret_y += self.offset_y_mm
        return ret_x, ret_y

if __name__ == "__main__":
    print(Transfrom2Machine(20,30))
