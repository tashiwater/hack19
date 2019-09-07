#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import pandas as pd
import math
from AR.pickselector import PickSelector

colorList = ['yellowgreen','gold','blue','pink','red','black','blown']

class BornDevider():
    def inputImage(self,img):
       
        bin_ = self.binarization(img)
        skel = self.getSkelton(bin_)
        self.img = skel
        raw_X, raw_y = self.imageToPoints(skel)
        inlier_X, inlier_y, coefficient = self.lineRANSAC(raw_X, raw_y, 10)
        self.X ,self.y = inlier_X, inlier_y
        return self.AignPoints(inlier_X,inlier_y,coefficient)
    
    def binarization(self, img):
        # Otsu's thresholding after Gaussian filtering
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(3,3),0)
        ret3,th3 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)
        #print(cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        reversed = cv2.bitwise_not(th3)
        
        return reversed
    
    def getSkelton(self,img):
        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)
        ret,img = cv2.threshold(img,100,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False

        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
        
            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True
        

        return skel

    def imageToPoints(self,img):
        height, width  = img.shape[:2]
        point_list = list()
        x_list = []
        y_list = []
        for x in range(width):
            for y in range(height):
                if img[y][x] == 255:
                    x_list.append(x)
                    y_list.append(y)####################################################################注意##############################
        X = np.array(x_list)
        y = np.array(y_list)
        raw_X = X.reshape(-1,1)
        raw_y = y.reshape(-1,1)
        
        return raw_X, raw_y

    def lineRANSAC(self, X, y, minPointNum):
        inlier_X  = []
        inlier_y  = []
        coefficient = []
        while len(X) > minPointNum:
            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor(max_trials=100)
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            
            inlier_X.append(X[inlier_mask])
            inlier_y.append(y[inlier_mask])

            a,b = self.reg1dim(X[inlier_mask],y[inlier_mask])
            coefficient.append([a,b])
            X = X[outlier_mask]
            y = y[outlier_mask]
        print("coefficient", coefficient)
        return inlier_X, inlier_y, coefficient
        

    def AignPoints(self, x_list, y_list,coefficient):
        n = len(coefficient)
        length = []
        pickSelector = PickSelector()
        for i in range(n):
            dot = x_list[i] * 1 +  y_list[i] * coefficient[i][0]
            size_ = math.sqrt(coefficient[i][0]*coefficient[i][0]+1)
            X = ( dot / (size_*size_) ) * 1
            Y = ( dot / (size_*size_) ) * coefficient[i][0]
            #print(X, Y)
            theta = math.atan2(coefficient[i][0],1)
            roll_X = X * math.cos(-theta) - Y * math.sin(-theta)
            roll_Y = X * math.sin(-theta) + Y * math.cos(-theta)
            #print(roll_X)
            length.append(roll_X.max()-roll_X.min())
            #print(x_list[i][np.argmax(roll_X)], y_list[i][np.argmax(roll_X)])
            #print(x_list[i][np.argmin(roll_X)], y_list[i][np.argmin(roll_X)])
            pickSelector.addPoint(x_list[i][np.argmax(roll_X)], y_list[i][np.argmax(roll_X)],x_list[i][np.argmin(roll_X)], y_list[i][np.argmin(roll_X)])
            #print("dist", roll_X.max()-roll_X.min())
            
            
            ###
            # for i in range(len(self.inlier_X)):
            #    plt.scatter(self.inlier_X[i], self.inlier_y[i], color=colorList[i], marker='.')
            #    plt.legend(loc='lower right')
            #    plt.xlabel("Input")
            #    plt.ylabel("Response")
            #plt.scatter(X, Y, color=colorList[2], marker='.')
            #plt.show()
        
        pick_points = pickSelector.calc()

        send_data = []
        for i in range(n):
            send_data.append([length[i], pick_points[i][0][0],pick_points[i][1][0]])  
        # print("send_data is",send_data)
        return send_data
            
    def dispPoints(self):
        for i in range(len(self.X)):
            plt.scatter(self.X[i], self.y[i], color=colorList[i], marker='.')
        plt.legend(loc='lower right')
        plt.xlabel("Input")
        plt.ylabel("Response")
        plt.show()

    def reg1dim(self, x, y):
        n = len(x)
        x = x.flatten()
        y = y.flatten()
        #print("sum: ",x.sum(),y.sum())
        a = ((np.dot(x, y)- y.sum() * x.sum()/n)/
            ((x ** 2).sum() - x.sum()**2 / n))
        b = (y.sum() - a * x.sum())/n
        #print("a, b",a,b)
        return a, b
    


if __name__ == "__main__":
    img = cv2.imread("img5.png",0)
    
    ########実装############################
    born = BornDevider()
    print(born.inputImage(img))
    ########################################

    born.dispPoints()


    #cv2.imshow("input",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
        
