#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:51:06 2020

@author: patricia
"""

import matplotlib.pyplot as plt
import numpy as np

roundabout = np.array([[392,309,405,309,393,297,407,297],
                       [368,313,383,313,368,300,382,300],
                       [365,315,381,315,366,301,380,301],
                       [346,320,365,320,346,305,364,304]])
crossing = np.array([[606,329,632,329,607,303,632,302],
                     [684,362,729,362,685,317,730,317],
                     [698,369,747,370,702,321,749,320],
                     [890,460,999,460,908,363,1008,364]])
giveway = np.array([[566,324,582,323,566,309,583,308],
                    [589,339,612,339,589,321,611,320],
                    [592,341,616,341,592,322,618,322],
                    [622,358,654,358,622,335,654,335]])

#print(roundabout[0,0])

def centroid(x0,y0,x1,y1,x2,y2,x3,y3):
    x = (x0+x1+x2+x3)/4
    y = (x0+y1+y2+y3)/4
    return np.array([x,y])

def movement(feature): #appends calculated centroids
    feature_c = [0 for x in range(2)] #initialize array
    for i in range(0,4):
        x0 = feature[i,0]
        x1 = feature[i,2]
        x2 = feature[i,4]
        x3 = feature[i,6]
        y0 = feature[i,1]
        y1 = feature[i,3]
        y2 = feature[i,5]
        y3 = feature[i,7]
        feature_c = np.vstack([feature_c,centroid(x0,y0,x1,y1,x2,y2,x3,y3)])
    return feature_c

roundabout_c = movement(roundabout)
crossing_c = movement(crossing)
giveway_c = movement(giveway)
#print(roundabout_c)

#TODO find matching point without given prior knowledge of detection
plt.plot([roundabout_c[1,0],roundabout_c[2,0],roundabout_c[3,0],roundabout_c[4,0]],[roundabout_c[1,1],roundabout_c[2,1],roundabout_c[3,1],roundabout_c[4,1]])
plt.plot([crossing_c[1,0],crossing_c[2,0],crossing_c[3,0],crossing_c[4,0]],[crossing_c[1,1],crossing_c[2,1],crossing_c[3,1],crossing_c[4,1]])
plt.plot([giveway_c[1,0],giveway_c[2,0],giveway_c[3,0],giveway_c[4,0]],[giveway_c[1,1],giveway_c[2,1],giveway_c[3,1],giveway_c[4,1]])
plt.xlim(0,1024)
plt.ylim(0,768)
