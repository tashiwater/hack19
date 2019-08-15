#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from machine.communicater import Communicater
from AR.transform import Transfrom2Machine
import os

if __name__ == "__main__":
    rospy.init_node('communicator', anonymous=True)
    matcher_srv = "match_srv"
    mbed_wait_topic = "from_mbed"
    parts_topic = "toCommunicator"
    mbed_go_topic = 'to_mbed'
    current_path = os.path.dirname(os.path.abspath(__file__))
    box_list_path = current_path + "/../data/box_list.csv"

    #座標変換用 param: machine座標系で見た画像座標系の原点の座標
    tf2machine = Transfrom2Machine(190, 180)

    communicater = Communicater(
        matcher_srv, mbed_wait_topic, parts_topic, mbed_go_topic,
        box_list_path, tf2machine)
    rospy.spin()

        
