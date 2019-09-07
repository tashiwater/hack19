#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import numpy as np
class PickSelector():
    def __init__(self):
        self.points = []

    def addPoint(self,x1,y1,x2,y2):
        tmp = [[x1,y1],[x2,y2]]
        self.points.append(tmp)
    
    def point_dist(self,x1,y1,x2,y2):
        return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        
    def calc(self):
        points = self.points
        #print(len(points))
        if len(points) == 1:
            x = (points[0][0][0] + points[0][1][0])/2
            y = (points[0][0][1] + points[0][1][1])/2
            tmp = []
            tmp.append([x,y])
            #print("return",tmp)
            return tmp
            
        return_data = []
        for i in range(len(points)):
            diff_data_start = []
            diff_data_goal = []
            for j in range(len(points)):

                if i != j:
                    diff_tmp_1 = self.point_dist(points[i][0][0],points[i][0][1],points[j][0][0],points[j][0][1])
                    diff_tmp_2 = self.point_dist(points[i][0][0],points[i][0][1],points[j][1][0],points[j][1][1])
                    diff_data_start.append(min(diff_tmp_1,diff_tmp_2))
                    #print(min(diff_tmp_1,diff_tmp_2))

                    diff_tmp_3 = self.point_dist(points[i][1][0],points[i][1][1],points[j][0][0],points[j][0][1])
                    diff_tmp_4 = self.point_dist(points[i][1][0],points[i][1][1],points[j][1][0],points[j][1][1])
                    diff_data_goal.append(min(diff_tmp_3,diff_tmp_4))
                    #print(min(diff_tmp_3,diff_tmp_4))
            start_min = np.array(diff_data_start).min()
            goal_min = np.array(diff_data_goal).min()
            #print(start_min,goal_min)
            if start_min > goal_min:
                return_data.append([points[i][0][0],points[i][0][1]])
                #print("start is min:" ,[points[i][0][0],points[i][0][1]])
            else:
                return_data.append([points[i][1][0],points[i][1][1]])
                #print("goal is min:" ,[points[i][1][0],points[i][1][1]])        
            
        #return_points.append(points[])
            
        #計算後, 初期化
        self.points = []
        #print("out points", return_data)
        return return_data