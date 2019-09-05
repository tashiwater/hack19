#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# import rospy
# from std_srvs.srv import SetBool, SetBoolResponse

import cv2
import xml.etree.ElementTree as ET
import os

from AR.contours import Contours


class FindParts():
    def __init__(self, testdata_path, tempsave_path,
                 locate_2d, get_frame_func):
        self.testdata_path = testdata_path
        self.save_path = tempsave_path
        self.locate_2d = locate_2d
        self.ar = self.locate_2d.ar_detect
        self.get_frame_func = get_frame_func
        self.ar_im = None
        print("locate_from_ar setup finish")

    def set_srv(self, find_parts_srv):
        rospy.Service(find_parts_srv, SetBool, self.srv_callback)

    def output_cont(self, output_dir_path):
        max_num = 0
        for filename in os.listdir(output_dir_path):
            num = int(filename[:-4])
            if num > max_num:
                max_num = num
        max_num += 1
        for i, show in enumerate(self.cont.show_cont()):
            cv2.imwrite(output_dir_path+"/" + str(i+max_num)+".png", show)

    def make_traindata(self, traindata_path):
        parts_name = raw_input("Input parts name:")
        output_path = traindata_path + "/" + parts_name
        if self.capture() is False:
            return False
        self.ar.find_marker()
        if self.get_roi() is False:
            return False
        self.find_contour()
        # ファイル作成
        if parts_name not in os.listdir(traindata_path):
            os.mkdir(output_path)
        self.output_cont(output_path)

    def get_testdata(self):
        if self.capture() is False:
            return False
        self.ar.find_marker()
        if self.get_roi() is False:
            return False
        self.find_contour()
        # self.output()
        return True

    def srv_callback(self, request):
        resp = SetBoolResponse()
        resp.success = self.get_testdata()
        resp.message = "called. data: " + str(request.data)
        print(resp.message)
        return resp

    def capture(self):
        # cap=cv2.VideoCapture(1)  # もともとなかった
        # cv2.waitKey(100)
        # _, frame = self.cap.read()
        # _, frame = cap.read()
        # cap.release()
        frame = self.get_frame_func()
        if frame is None:
            print("there is no img")
            return False
        self.ar.img = frame
        cv2.imwrite(self.save_path + "/raw.png", self.ar.img)
        return True

    def get_roi(self):
        self.ar_im = self.ar.get_ar_detect_img()
        # small_im = cv2.resize(ar_im, None, fx = 0.5, fy = 0.5)
        # self.show_img("ar", ar_im)
        self.roi_img = self.locate_2d.get_roi_img()
        if self.roi_img is None:
            print("there is no roi img")
            return False
        return True

    def find_contour(self):
        self.cont = Contours(self.roi_img)
        cnt_img = self.cont.find_contours()
        return cnt_img

    def show_img(self, name, img, width=400):
        scale = width / img.shape[1]
        show = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.imshow(name, show)
        cv2.waitKey(200)
        cv2.imwrite(self.save_path + "/" + name + ".png", img)

    def output(self):
        self.cleanFiles()
        for i, (show, at) in enumerate(zip(self.cont.show_cont(), self.cont.get_at())):
            posi = self.locate_2d.pred_posi_in_roi(at)
            print(posi)
            # cv2.imshow("find" + str(i), show)
            cv2.imwrite(self.testdata_path + "/imgs/img" + str(i)+".png", show)
            self.make_xml(i, posi)

    def make_xml(self, i, posi):
        parts = ET.Element('parts')
        x = ET.SubElement(parts, "x_m")
        x.text = str(posi[0])
        y = ET.SubElement(parts, "y_m")
        y.text = str(posi[1])
        tree = ET.ElementTree(parts)
        tree.write(self.testdata_path + '/xmls/' + str(i) + '.xml',
                   encoding="UTF-8")

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

    def get_diagonal_mm(self):
        ret = []
        for px in self.cont.get_diagonal_px():
            temp = [px, 0]
            lenght = self.locate_2d.pred_posi_in_roi(temp)[0]
            ret.append(lenght)
        return ret

    def get_neji_output(self):
        return [[length, self.locate_2d.pred_posi_in_roi(at), img] for length, at, img in zip(self.get_diagonal_mm(), self.cont.get_at(), self.cont.show_cont())]
