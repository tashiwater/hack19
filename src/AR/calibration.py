# -*- coding: utf-8 -*-

import numpy
import cv2
# from glob import glob
import Tkinter
import tkMessageBox
import rospy

square_size = 0.24     # 正方形のサイズ
pattern_size = (10, 7)  # 模様のサイズ
pattern_points = numpy.zeros( (numpy.prod(pattern_size), 3), numpy.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = numpy.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
obj_points = []
img_points = []
print(pattern_points)
cap = cv2.VideoCapture(2) #ビデオキャプチャの開始
cap.set(3, 1280)
cap.set(4, 720)
# for i in glob("*.jpg"):
for i in range(100):
    if rospy.is_shutdown():
        break
    fn = str(i)
    # 画像の取得
    # im = cv2.imread(fn, 0)
    for i in xrange(5):
        _, frame = cap.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print "loading..." + fn
    # チェスボードのコーナーを検出
    found, corner = cv2.findChessboardCorners(im, pattern_size)
    cv2.imshow("img",im)
    # コーナーがあれば
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im, corner, (5,5), (-1,-1), term)    #よくわからないがサブピクセル処理（小数点以下のピクセル単位まで精度を求める）
        cv2.drawChessboardCorners(im, pattern_size, corner,found)
        cv2.imshow('found corners in' + fn,im)
    # コーナーがない場合のエラー処理
    cv2.waitKey(1000)
    if not found:
        print 'chessboard not found'
        continue
    # 選択ボタンを表示
    root = Tkinter.Tk()
    root.withdraw()
    if tkMessageBox.askyesno('askyesno','この画像の値を採用しますか？'):
        img_points.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加 #corner.reshape(-1, 2) : 検出したコーナーの画像内座標値(x, y)
        obj_points.append(pattern_points)
        print 'found corners in ' + fn + ' is adopted'
    else:
        print 'found corners in　' + fn + ' is not adopted'  
    cv2.destroyAllWindows()
    if len(img_points) > 10:
        break
cap.release()
    
# 内部パラメータを計算
rms, cameraMatrix, distCoeffs, r, t = cv2.calibrateCamera(obj_points,img_points,(im.shape[1],im.shape[0]),None,None)
# 計算結果を表示
print "RMS = ", rms
print "K = \n", cameraMatrix
print "d = ", distCoeffs.ravel()
# 計算結果を保存
numpy.savetxt("cameraMatrix.csv", cameraMatrix, delimiter =',',fmt="%0.14f") #カメラ行列の保存
numpy.savetxt("distCoeffs.csv",distCoeffs, delimiter =',',fmt="%0.14f") #歪み係数の保存
