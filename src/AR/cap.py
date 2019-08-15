#!usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def get_img():
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    img = None
    while True:
        _, img = cap.read()#カメラの画像を読み込む
        cv2.imshow("pc camera", img) #ウィンドウに画像を出力
        #Enterが押されたらループを抜ける
        print("loop")
        k = cv2.waitKey(200)#1ms確認
        if k == 13:
            break
    cap.release()
    cv2.destroyAllWindows()
    return img

class Subscriber():
    def __init__(self, topic, type_name):
        rospy.Subscriber(topic, type_name, self.call_back)
        self.called = False
    def call_back(self, input):
        self.value = input
        self.called =True


if __name__ == "__main__":
    rospy.init_node("capture_node")
    # rospy.Publisher("camera_img", Image)
    sub_img = Subscriber("/camera/color/image_raw", Image)
    bridge = CvBridge()
    while sub_img.called == False:
        pass

    while not rospy.is_shutdown():
        img = bridge.imgmsg_to_cv2(sub_img.value,"bgr8") 
        cv2.imwrite("temp.png", img)
        cv2.waitKey(100)

    # while 
    # # img = get_img()
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()