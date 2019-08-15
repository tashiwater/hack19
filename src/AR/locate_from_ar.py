#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from AR.ar_detect import ARDetect
import numpy as np
from cv2 import aruco
import cv2
from contours import Contours
import xml.etree.ElementTree as ET
from AR.locate2d import Locate2d
import os

current_path = os.path.dirname(os.path.abspath(__file__))
testdata_path = current_path + "/../data/test"
def make_xml(i, posi):
    parts = ET.Element('parts')
    x = ET.SubElement(parts, "x_m")
    x.text = str(posi[0])
    y = ET.SubElement(parts, "y_m")
    y.text = str(posi[1])
    tree = ET.ElementTree(parts)
    tree.write(testdata_path + '/xmls/' + str(i) + '.xml', encoding="UTF-8")


def cleanFiles():
    xml_dirs = os.listdir(current_path+"/xmls")
    img_dirs = os.listdir(current_path+"/imgs")
    if len(xml_dirs) > 0:
        for f in xml_dirs:
            print("remove:", f)
            os.remove(current_path+'/xmls/'+f)
    else:
        print("no pre files")
    if len(img_dirs) > 0:
        for f in img_dirs:
            print("remove:", f)
            os.remove(current_path+'/imgs/'+f)
    else:
        print("no pre files")


if __name__ == "__main__":
    
    ar_marker_size = 0.02  #ARマーカー一辺[m]
    camera_matrix = np.loadtxt(current_path +"/cameraMatrix.csv", delimiter= ",")
    distCoeffs = np.loadtxt(current_path +"/distCoeffs.csv", delimiter= ",")
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
        frame = cv2.imread("/mnt/c/Users/shimi/Pictures/Camera Roll/temp.jpg")
        if frame is None:
            continue
        ar.img = frame          
        #ARマーカー検出
        ar.find_marker()
        ar.get_corner(0)
        ar_im = ar.get_ar_detect_img()
        small_im = cv2.resize(ar_im, None, fx = 0.5, fy = 0.5)
        cv2.imshow("ar", small_im)
        roi_img = locate_2d.get_roi_img()
        if roi_img is None:
            continue
        cv2.namedWindow("roi",  cv2.WINDOW_NORMAL)
        cv2.imshow("roi", roi_img)
        cont = Contours(roi_img)
        cont.find_contours()

        i = 0
        
        cleanFiles()
        for show, at in zip(cont.show_cont(),cont.get_at()) :
            posi = locate_2d.pred_posi_in_roi(at)
            print(posi)
            cv2.imshow("find" + str(i), show)
            cv2.imwrite(testdata_path + "/imgs/img"+ str(i)+".png", show)
            make_xml(i, posi)
            i += 1
            cv2.waitKey(100)
        print("finish")
        break
    # cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
