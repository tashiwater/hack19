#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import cv2
import numpy as np

import xml.etree.ElementTree as ET

class RecognizedObject():
    def __init__(self, path):
        self.tree = ET.parse(path)
        root = self.tree.getroot()
        self.x_m = float(root[0].text)
        self.y_m = float(root[1].text)
        name = os.path.basename(path)
        self.name,ext = os.path.splitext(name)
        xml_dir_path = os.path.dirname(path)
        image = cv2.imread(xml_dir_path+'/../imgs/img'+self.name+'.png')
        self.image = image
        # cv2.imshow("img", self.image)
        # cv2.waitKey(0)
        # image = cv2.resize(image, (28,28))
        # self.image_np = image.flatten().astype(np.float32)/255.0


    def dispStates(self):
        print(self.x_m , ",", self.y_m)

    def showImage(self):
        cv2.imshow("img",self.image)
        cv2.waitKey(0)

if __name__ == "__main__":
    xml_dirs = os.listdir(current_path+"/AR/xmls")
    if len(xml_dirs) > 0:
        #オブジェクトデータの取得 
        for f in xml_dirs:        
            obj = RecognizedObject(current_path+'/AR/xmls/' + f)
            objs.append(obj)
    
