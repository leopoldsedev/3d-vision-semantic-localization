#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:51:06 2020

@author: patricia
"""
import matplotlib.pyplot as plt
import numpy as np
import math


#by class
roundabout = np.array([[392,309,405,309,393,297,407,297],
                       [368,313,383,313,368,300,382,300],
                       [365,315,381,315,366,301,380,301],
                       [363,315,379,316,363,302,379,301]])
#                       [357,316,373,317,356,302,372,301]
#                       [346,320,365,320,346,305,364,304]
crossing = np.array([[606,329,632,329,607,303,632,302],
                     [684,362,729,362,685,317,730,317],
                     [698,369,747,370,702,321,749,320],
                     [716,377,768,378,718,325,771,324]])
#                     [773,402,840,403,777,336,844,338]
#                     [890,460,999,460,908,363,1008,364]
giveway = np.array([[566,324,582,323,566,309,583,308],
                    [589,339,612,339,589,321,611,320],
                    [592,341,616,341,592,322,618,322],
                    [597,344,621,344,596,322,621,324]])
#                    [607,347,634,349,606,326,635,325]
#                    [622,358,654,358,622,335,654,335]

#by image
img1 = np.vstack((roundabout[0,:],crossing[0,:],giveway[0,:])) #img_CAMERA1_1261229993.980124_right.jpg
img2 = np.vstack((roundabout[1,:],crossing[1,:],giveway[1,:])) #img_CAMERA1_1261229994.980144_right.jpg
img3 = np.vstack((roundabout[2,:],crossing[2,:],giveway[2,:])) #img_CAMERA1_1261229995.080157_right.jpg
img4 = np.vstack((roundabout[3,:],crossing[3,:],giveway[3,:])) #img_CAMERA1_1261229995.730161_right.jpg


#print(roundabout[0,:])

def centroid(x0,y0,x1,y1,x2,y2,x3,y3):
    x = (x0+x1+x2+x3)/4
    y = (x0+y1+y2+y3)/4
    return np.array([x,y])

def distance(p1,p2): #p = [x,y]
    dist = math.sqrt((p1[0]-p1[1])^2+(p2[0]-p2[1])^2)
    return dist #distance between two points

#test = distance([1,0],[0,1])
#print(test)

def movement(feature): #sequence of calculated centroids
    feature_c = [0 for x in range(2)] #initialize array
    for i in range(0,4):#4 images for now
        x0 = feature[i,0]
        x1 = feature[i,2]
        x2 = feature[i,4]
        x3 = feature[i,6]
        y0 = feature[i,1]
        y1 = feature[i,3]
        y2 = feature[i,5]
        y3 = feature[i,7]
        feature_c = np.vstack([feature_c,centroid(x0,y0,x1,y1,x2,y2,x3,y3)])
    return feature_c #sequence of calculated centroids

def distribution(img):
    c=[0,0]
    for i in range(0,3):
        c = np.vstack([c,centroid(img[i,0],img[i,2],img[i,4],img[i,6],img[i,1],img[i,3],img[i,5],img[i,7])])
    return c #centroid of each feature in a picture

img1_c = distribution(img1)
img2_c = distribution(img2)
img3_c = distribution(img3)
img4_c = distribution(img4)

#TODO this function should return matches that can be put in movement
#create dist as distances stacked
#find the minimum -> true
def match(img_prev,img):
    [num_feat1,dim] = img_prev.shape
    [num_feat2,dim] = img.shape
    pairs = [0,0]
    for i in range(0,num_feat1):
        dist = []
        for j in range(0,num_feat1):
            dist.append(distance(img_prev[i,:],img[j,:]))
        print(dist)
        matched = np.argmin(dist)
        pair = [i,matched] #row of img_prev, row of img
        pairs = np.vstack([pairs,pair])
    print("[col_prev,col]",pairs)
    return pairs


test1 = match(img3,img4)
test2 = match(img1,img2)
feat1 = np.vstack([[img1[test1[1,0]],img2[test1[1,1]]],
         [img3[test2[1,0]],img4[test2[1,1]]]]) #reference the second row of pairs
feat2 = np.vstack([[img1[test1[2,0]],img2[test1[2,1]]],
         [img3[test2[2,0]],img4[test2[2,1]]]])
feat3 = np.vstack([[img1[test1[3,0]],img2[test1[3,1]]],
         [img3[test2[3,0]],img4[test2[3,1]]]])

#print(feat1.shape)
#print(giveway)

#roundabout_c = movement(roundabout)
#crossing_c = movement(crossing)
#giveway_c = movement(giveway)

feat1_c = movement(feat1)
feat2_c = movement(feat2)
feat3_c = movement(feat3)


#plt.plot([roundabout_c[1,0],roundabout_c[2,0],roundabout_c[3,0],roundabout_c[4,0]],[roundabout_c[1,1],roundabout_c[2,1],roundabout_c[3,1],roundabout_c[4,1]])
#plt.plot([crossing_c[1,0],crossing_c[2,0],crossing_c[3,0],crossing_c[4,0]],[crossing_c[1,1],crossing_c[2,1],crossing_c[3,1],crossing_c[4,1]])
#plt.plot([giveway_c[1,0],giveway_c[2,0],giveway_c[3,0],giveway_c[4,0]],[giveway_c[1,1],giveway_c[2,1],giveway_c[3,1],giveway_c[4,1]])

plt.plot([feat1_c[1,0],feat1_c[2,0],feat1_c[3,0],feat1_c[4,0]],[feat1_c[1,1],feat1_c[2,1],feat1_c[3,1],feat1_c[4,1]])
plt.plot([feat2_c[1,0],feat2_c[2,0],feat2_c[3,0],feat2_c[4,0]],[feat2_c[1,1],feat2_c[2,1],feat2_c[3,1],feat2_c[4,1]])
plt.plot([feat3_c[1,0],feat3_c[2,0],feat3_c[3,0],feat3_c[4,0]],[feat3_c[1,1],feat3_c[2,1],feat3_c[3,1],feat3_c[4,1]])

plt.xlim(0,1024)
plt.ylim(0,768)
