#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import rospy

import os

from machine.matching import Match

if __name__ == "__main__":
    rospy.init_node("matching_node")

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../data"
    checkpoint_path = data_path+"/checkpoint/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    testdata_path = current_path + "/../data/test"
    pub_topic = 'toCommunicator'
    match_srv = "match_srv"
    get_target_srv = "get_target_srv"
    match = Match(data_path, testdata_path, pub_topic,
                  match_srv, get_target_srv)
    rospy.spin()
