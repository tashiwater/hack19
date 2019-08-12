#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool


class Subscriber():
    def __init__(self, topic, type_name):
        rospy.Subscriber(topic, type_name, self.call_back)
        self.called = False

    def call_back(self, input):
        self.value = input
        self.called = True

        

if __name__ == "__main__":
    pub_target = rospy.Publisher('to_mbed',Quaternion,queue_size=10)
    pub_go_sign = rospy.Publisher('to_locate', Bool,queue_size=10)
    rospy.init_node('communicator', anonymous=True)
    sub_target = Subscriber("toCommunicator", Vector3)
    sub_go_sign = Subscriber("from_mbed", Bool)

    while not rospy.is_shutdown():
        if sub_go_sign.called == False:
            print("wait for go sign")
            continue
        sub_go_sign.called = False
        #ここに処理
        pub_go_sign.publish(sub_go_sign.value)
        while not rospy.is_shutdown():
            if sub_target == True:
                break
            print("wait for target")
        pub_target.publish(sub_target.value)
            

        