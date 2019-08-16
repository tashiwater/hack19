#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import rospy
from std_srvs.srv import SetBool, SetBoolResponse

import cv2
import xml.etree.ElementTree as ET
import os

from AR.contours import Contours
class FindParts():
    def __init__(self, camera_info_path, testdata_path, tempsave_path,
                 locate_2d, cap, find_parts_srv, raw_img_path):
        self.testdata_path = testdata_path
        self.save_path = tempsave_path
        self.raw_img_path = raw_img_path
        self.locate_2d = locate_2d
        self.ar = self.locate_2d.ar_detect
        self.cap = cap
        rospy.Service(find_parts_srv, SetBool, self.srv_callback)
        print("locate_from_ar setup finish")


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
        return resp

    def capture(self):
        # _, frame = self.cap.read()
        frame = cv2.imread(self.raw_img_path)
        if frame is None:
            print("there is no img")
            return False
        self.ar.img = frame
        cv2.imwrite(self.save_path + "/raw.png", self.ar.img)
        return True
    
    def get_roi(self):
        ar_im = self.ar.get_ar_detect_img()
        # small_im = cv2.resize(ar_im, None, fx = 0.5, fy = 0.5)
        # cv2.imshow("ar", small_im)
        cv2.imwrite(self.save_path + "/ar_img.png", ar_im)
        self.roi_img = self.locate_2d.get_roi_img()
        if self.roi_img is None:
            print("there is no roi img")
            return None
        # cv2.namedWindow("roi",  cv2.WINDOW_NORMAL)
        cv2.imwrite(self.save_path + "/roi.png", self.roi_img)
        # cv2.imshow("roi", roi_img)
        return True

    def find_contour(self):
        self.cont = Contours(self.roi_img)
        self.cont.find_contours()

    def output(self):
        self.cleanFiles()
        for i, (show, at) in enumerate(zip(self.cont.show_cont(),self.cont.get_at())) :
            posi = self.locate_2d.pred_posi_in_roi(at)
            print(posi)
            # cv2.imshow("find" + str(i), show)
            cv2.imwrite(self.testdata_path + "/imgs/img"+ str(i)+".png", show)
            self.make_xml(i, posi)
    
    def make_xml(self, i, posi):
        parts=ET.Element('parts')
        x=ET.SubElement(parts, "x_m")
        x.text=str(posi[0])
        y=ET.SubElement(parts, "y_m")
        y.text=str(posi[1])
        tree=ET.ElementTree(parts)
        tree.write(self.testdata_path + '/xmls/' + str(i) + '.xml',
         encoding = "UTF-8")

    def cleanFiles(self):
        xml_dirs = os.listdir(self.testdata_path+"/xmls")
        img_dirs = os.listdir(self.testdata_path+"/imgs")
        if len(xml_dirs) > 0:
            for f in xml_dirs:
                print("remove:", f)
                os.remove(self.testdata_path+'/xmls/'+f)
        else:
            print("no pre files")
        if len(img_dirs) > 0:
            for f in img_dirs:
                print("remove:", f)
                os.remove(self.testdata_path+'/imgs/'+f)
        else:
            print("no pre files")

