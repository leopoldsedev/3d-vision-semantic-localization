#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:51:06 2020

@author: patricia
"""

import matplotlib.pyplot as plt
import numpy as np

roundabout = np.array([[392,309,40,309,393,297,407,297],
                       [368,313,383,313,368,300,382,300]])
crossing = np.array([[606,329,632,329,607,303,632,302],
                     [684,362,729,362,685,317,730,317]])

#print(roundabout[0,0])

def centroid(x0,y0,x1,y1,x2,y2,x3,y3):
    x = (x0+x1+x2+x3)/4
    y = (x0+y1+y2+y3)/4
    return np.array([x,y])

def movement(feature): #appends calculated centroids
    feature_c = [0 for x in range(2)] #initialize array
    for i in range(0,2):
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
print(roundabout_c)


plt.plot([roundabout_c[1,0],roundabout_c[2,0]],[roundabout_c[1,1],roundabout_c[2,1]])
plt.xlim(0,1024)
plt.ylim(0,768)
