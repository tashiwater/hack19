#!usr/bin/env python3
# -*- coding: utf-8 -*-
from cv2 import aruco
import cv2
import numpy as np
class ARDetect():
    def __init__(self, marker_length, ar_dictionary,camera_matrix,distort_coeff ):
        self.marker_length = marker_length
        self.dictionary = aruco.getPredefinedDictionary(ar_dictionary)
        self.camera_matrix = camera_matrix
        self.distort_coeff = distort_coeff
    
    def find_marker(self):
        self.corners, self.ids, self.rejectedImgPoints = \
            aruco.detectMarkers(self.img,self.dictionary)

    def get_ar_detect_img(self):
        im_copy = self.img.copy()
        aruco.drawDetectedMarkers(im_copy, self.corners, self.ids, (0,255,0))
        return im_copy

    def show(self):
        im = self.get_ar_detect_img()
        cv2.imshow("result", im)
        
    def get_ar_posi(self, id):
        corner = self.get_corner(id)
        return self.get_posi(corner)

    def get_posi(self, corner):
        if len(corner)  < 1:
            return [], []
        ret = aruco.estimatePoseSingleMarkers([np.array(corner, dtype = float)],self.marker_length, self.camera_matrix, self.distort_coeff)
        rvec, tvec = ret[0], ret[1]
        return rvec[0][0], tvec[0][0]

    def marker_num(self):
        return len(self.corners)
    
    def get_id_index(self,id):
        index = np.where(self.ids == id)[0]
        if len(index) < 1:
            return None
        return index[0]

    def get_corner(self, id):
        index = self.get_id_index(id)
        if index == None:
            return []
        ret = self.corners[index][0]
        return ret

if __name__ == "__main__":
    ar_marker_size = 0.08 #ARマーカー一辺[m]
    camera_matrix = np.loadtxt("cameraMatrix.csv", delimiter= ",")
    distCoeffs = np.loadtxt("distCoeffs.csv", delimiter= ",")
    ar = ARDetect(ar_marker_size, aruco.DICT_4X4_50, camera_matrix, distCoeffs)
    cap = cv2.VideoCapture(0)
    print("capture")
    while True:
        print("while")
        _, frame = cap.read()
        print("read")
        ar.img = frame          
        #ARマーカー検出
        ar.find_marker()
        ar.show()
        # ar.get
        rvec, posis = ar.get_ar_posi(0)
        print(posis)
        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()

