#!usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt

def get_img(name):
    while True:
        img = cv2.imread(name)
        if img is None:
            continue
        return img

class Contours():
    def __init__(self, img):
        self.raw_img = img.copy()
        self.img = img.copy()
    
    def find_contours(self):
        img = self.img
        #二値化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #画像を平滑化(白色雑音の除去). ksize:フィルタ窓枠の大きさ, sigmaX: ガウシアンの標準偏差値. 0だとカーネルのサイズから自動的に計算
        graygauss = cv2.GaussianBlur(gray , ksize = (3,3),sigmaX =  0)
        # 二値化. thresh:閾値.  maxval:閾値以上(指定により閾値以下のこともある)の値を持つ画素に対して割り当てられる値,
        # type:二値化の方法。cv2.THRESH_BINARY_INVだと白黒かつ反転
        # 返り値:[1]に画像が入っている
        im2 = cv2.threshold(graygauss, thresh = 100, maxval = 240, type = cv2.THRESH_BINARY_INV)[1]
        
        # cv2.imshow("threshold", im2)
        # plt.imshow(im2, cmap="gray")
        # cv2.waitKey(0)
        #輪郭抽出
        cnts = cv2.findContours(im2, mode = cv2.RETR_EXTERNAL, 
                                method = cv2.CHAIN_APPROX_SIMPLE)[1]
        #輪郭図示
        red = (0,0,255)
        size_list = []
        for pt in cnts:
            #輪郭を含む長方形を作る
            x,y,w,h = cv2.boundingRect(pt)
            # 小さい領域はスルー
            if w < 20 or h < 20:
                continue
            size_list.append((x,x + w,y, y+h))
            #長方形を図示
            cv2.rectangle(img,(x,y), (x+w,y+h), red, thickness =10)
        cv2.imshow("contours", img)
        self.size_list = size_list

    def show_cont(self):
        for lis in self.size_list:
            temp =  self.raw_img[lis[2]:lis[3], lis[0]:lis[1]]
            yield temp
    
    def get_corner(self):
        for lis in self.size_list:
            temp = [[lis[0], lis[2]],
                    [lis[1], lis[2]],
                    [lis[1], lis[3]],
                    [lis[0], lis[3]],
                    ]
            yield temp
    
    #領域の中心画素を返す
    def get_at(self):
        for lis in self.size_list:
            temp = [(lis[0] + lis[1]) * 0.5, (lis[2] + lis[3]) * 0.5]
            yield temp




if __name__ == "__main__":
    #PCカメラから入力. 0がデフォルト。自分は0で前面,1で背面カメラが作動 
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 1280)
    # cap.set(4, 720)
        # _, img = cap.read()#カメラの画像を読み込む
        # cv2.imshow("pc camera", img) #ウィンドウに画像を出力
        #Enterが押されたらループを抜ける
        # k = cv2.waitKey(100)#1ms確認
        # if k == 13:
        #     break
    # cap.release()#カ   i = 0メラ解放
    # cv2.destroyAllWin    for show, corner in zip(cont.show_cont(),cont.get_corner()) :dows() # window破棄
        # cv2.imwrite("./neji_40/"+str(i) + "output.png", gen)
    img = get_img("temp1.png")
    cv2.imshow("raw", img)
    cont = Contours(img)
    cont.find_contours()
    
    i = 0
    for show, corner in zip(cont.show_cont(),cont.get_corner()) :
        # cv2.imwrite("./neji_40/"+str(i) + "output.png", gen)
        i+=1
        print(corner)
        cv2.imshow("find", show)
        cv2.waitKey(0)
