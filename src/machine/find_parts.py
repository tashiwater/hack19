#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from ar_detect import ARDetect
import numpy as np
from cv2 import aruco
import cv2
from contours import Contours
import xml.etree.ElementTree as ET
from locate2d import Locate2d
import os
from std_srvs.srv import SetBool, SetBoolResponse
import rospy

class FindParts():
    def __init__(self):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        ar_marker_size=0.02  # ARマーカー一辺[m]
        camera_matrix=np.loadtxt(
    self.current_path + "/cameraMatrix.csv", delimiter = ",")
        distCoeffs=np.loadtxt(
    self.current_path + "/distCoeffs.csv", delimiter = ",")
        self.ar=ARDetect(ar_marker_size, aruco.DICT_4X4_50,
    camera_matrix, distCoeffs)
        print("locate_from_ar setup")
        self.locate_2d=Locate2d(self.ar, 0.2, 0.159, 0.017, 0.005)
        cap=cv2.VideoCapture(0)  # もともとなかった
        cap.set(3, 1280)
        cap.set(4, 720)
        find_parts_srv=rospy.get_param("~find_parts_srv", "find_parts_srv")
        rospy.Service(find_parts_srv, SetBool, self.srv_callback)

    def srv_callback(self, request):
        resp=SetBoolResponse()
        resp.success=True
        resp.message="called. data: " + str(request.data)
        print(resp.message)

        if self.capture() is False:
            return resp
        self.ar.find_marker()
        if self.get_roi() is False:
            return resp
        self.find_contour()
        self.output()


    def capture(self):
        frame=cv2.imread("/mnt/c/Users/shimi/Pictures/Camera Roll/temp.jpg")
        if frame is None:
            return False
        self.ar.img = frame
        return True
    
    def get_roi(self):
        self.ar.get_corner(0)
        ar_im = self.ar.get_ar_detect_img()
        small_im = cv2.resize(ar_im, None, fx = 0.5, fy = 0.5)
        cv2.imshow("ar", small_im)
        roi_img = self.locate_2d.get_roi_img()
        if roi_img is None:
            return None
        cv2.namedWindow("roi",  cv2.WINDOW_NORMAL)
        cv2.imshow("roi", roi_img)
        return True

    def find_contour(self):
        self.cont = Contours(roi_img)
        self.cont.find_contours()

    def output(self):
        for i, (show, at) in enumerate(zip(self.cont.show_cont(),self.cont.get_at())) :
            posi = self.locate_2d.pred_posi_in_roi(at)
            print(posi)
            cv2.imshow("find" + str(i), show)
            cv2.imwrite(self.current_path + "/imgs/img"+ str(i)+".png", show)
            make_xml(i, posi)
    
    def make_xml(i, posi):
        parts=ET.Element('parts')
        x=ET.SubElement(parts, "x_m")
        x.text=str(posi[0])
        y=ET.SubElement(parts, "y_m")
        y.text=str(posi[1])
        tree=ET.ElementTree(parts)
        tree.write(self.current_path + '/xmls/' + str(i) + '.xml', encoding = "UTF-8")
