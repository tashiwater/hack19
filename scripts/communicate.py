#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
import pandas as pd
from transform import Transfrom2Machine
import os
class Subscriber():
    def __init__(self, topic, type_name):
        rospy.Subscriber(topic, type_name, self.call_back)
        self.called = False

    def call_back(self, input):
        self.value = input
        self.called = True

if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    pub_target = rospy.Publisher('to_mbed',Quaternion,queue_size=1)
    pub_go_sign = rospy.Publisher('to_locate', Bool,queue_size=1)
    rospy.init_node('communicator', anonymous=True)
    sub_target = Subscriber("toCommunicator", Vector3)
    sub_go_sign = Subscriber("from_mbed", Bool)
    box_df = pd.read_csv(current_path + "/box_list.csv", header=0)
    #座標変換用 param: machine座標系で見た画像座標系の原点の座標
    tf2machine = Transfrom2Machine(190, 180) 
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        if sub_go_sign.called == False:
            print("wait for go sign")
            continue
        sub_go_sign.called = False
        pub_go_sign.publish(sub_go_sign.value)
        print("wait for target")
        while not rospy.is_shutdown():
            if sub_target.called == True:
                break
        sub_target.called = False
        print(sub_target.value)
        to_mbed = Quaternion()
        target = sub_target.value
        to_mbed.x, to_mbed.y = tf2machine.get_xy_mm(target.x, target.y)
        id = int(target.z)
        box = box_df.iloc[id]
        to_mbed.z = box.box_x
        to_mbed.w = box.box_y
        print("pub to mbed ", to_mbed)
        pub_target.publish(to_mbed)

        
