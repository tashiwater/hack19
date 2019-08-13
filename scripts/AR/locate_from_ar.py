#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from ar_detect import ARDetect
import numpy as np
from cv2 import aruco
import cv2
from contours import Contours
import xml.etree.ElementTree as ET
from locate2d import Locate2d

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
