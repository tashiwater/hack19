#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
import pandas as pd
from std_srvs.srv import SetBool

class Communicater():
    def __init__(self, matcher_srv, mbed_wait_topic, parts_topic, mbed_go_topic, box_list_path, 
                 tf2machine, get_target_srv, find_parts_srv):
        rospy.Subscriber(mbed_wait_topic, Bool, self.from_mbed_cb)
        rospy.Subscriber(parts_topic, Vector3, self.match_cb)
        self.pub_target = rospy.Publisher(mbed_go_topic, Quaternion, queue_size=1)
        self.matcher_srv_call = None
        rospy.wait_for_service(matcher_srv)
        self.matcher_srv_call = rospy.ServiceProxy(matcher_srv, SetBool)
        rospy.wait_for_service(get_target_srv)
        self.get_target_srv_call = rospy.ServiceProxy(get_target_srv, SetBool)
        rospy.wait_for_service(find_parts_srv)
        self.find_parts_srv_call = rospy.ServiceProxy(find_parts_srv, SetBool)
        self.box_df = pd.read_csv(box_list_path,header=0)
        self.tf2machine = tf2machine

    def from_mbed_cb(self, data):
        if self.matcher_srv_call is None or self.get_target_srv_call is None:
            print("no service")
            return
        try:
            print("call")
            self.find_parts_srv_call(True)
            print("called")
            self.matcher_srv_call(True)
            self.get_target_srv_call(True)
        except rospy.ServiceException, e:
            print ("Service call failed: %s" % e)

    def match_cb(self, value):
        to_mbed = Quaternion()
        target = value
        to_mbed.x, to_mbed.y = self.tf2machine.get_xy_mm(target.x, target.y)
        id = int(target.z)
        box = self.box_df.iloc[id]
        to_mbed.z = box.box_x
        to_mbed.w = box.box_y
        print("pub to mbed ", to_mbed)
        self.pub_target.publish(to_mbed)
