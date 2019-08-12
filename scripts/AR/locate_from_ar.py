#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from ar_detect import ARDetect
import numpy as np
from cv2 import aruco
import cv2
from contours import Contours
import xml.etree.ElementTree as ET

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


def make_xml(i, posi):
    parts = ET.Element('parts')
    x = ET.SubElement(parts, "x_m")
    x.text = str(posi[0])
    y = ET.SubElement(parts, "y_m")
    y.text = str(posi[1])
    tree = ET.ElementTree(parts)
    tree.write('./xmls/' + str(i) + '.xml', encoding="UTF-8")


if __name__ == "__main__":
    ar_marker_size = 0.02 #ARマーカー一辺[m]
    camera_matrix = np.loadtxt("cameraMatrix.csv", delimiter= ",")
    distCoeffs = np.loadtxt("distCoeffs.csv", delimiter= ",")
    ar = ARDetect(ar_marker_size, aruco.DICT_4X4_50, camera_matrix, distCoeffs)
    print("locate_from_ar setup")
    locate_2d = Locate2d(ar, 0.2, 0.159, 0.017, 0.005)
    cap = cv2.VideoCapture(0)##もともとなかった
    cap.set(3, 1280)
    cap.set(4, 720)
    while True:
        cv2.waitKey(200)
        #if cv2.waitKey(10) > 0:
        #    break
        # _, frame = cap.read()
        frame = cv2.imread("temp.jpg")
        if frame is None:
            continue
        ar.img = frame          
        #ARマーカー検出
        ar.find_marker()
        ar.get_corner(0)
        ar.show()
        # print(posis)
        roi_img = locate_2d.get_roi_img()
        if roi_img is None:
            continue
        cv2.namedWindow("roi",  cv2.WINDOW_NORMAL)
        cv2.imshow("roi", roi_img)
        cont = Contours(roi_img)
        cont.find_contours()

        i = 0
        

        for show, at in zip(cont.show_cont(),cont.get_at()) :
            posi = locate_2d.pred_posi_in_roi(at)
            print(posi)
            cv2.imshow("find", show)
            cv2.imwrite("./imgs/img"+ str(i)+".png", show)
            make_xml(i, posi)
            i += 1
            cv2.waitKey(0)

    # cap.release()
    cv2.destroyAllWindows()
