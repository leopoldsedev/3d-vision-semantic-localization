import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
import sys
import matplotlib.image as mpimg

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from pykalman import KalmanFilter

import cv2
import camera
# import get_position_ground_truth from 

np.set_printoptions(threshold=100, linewidth=200)


gps_full_data = np.genfromtxt('data/07/gps.csv', skip_header=1)

# TODO GPS data also includes speed and heading
gps_timestamps = gps_full_data[:,0]
gps_geocen_x = gps_full_data[:,12]
gps_geocen_y = gps_full_data[:,13]
gps_geocen_z = gps_full_data[:,14]
gps_geocen = np.block([[gps_geocen_x], [gps_geocen_y], [gps_geocen_z]]).T # size n x 3
gps_local_x = gps_full_data[:,8]
gps_local_y = gps_full_data[:,9]
gps_local_z = gps_full_data[:,10]
gps_local = np.block([[gps_local_x], [gps_local_y], [gps_local_z]]).T # size n x 3
# Good image sequence from extract #07 -- 450 to 800 (right)
# start: img_CAMERA1_1261229992.780127_right.jpg (index 450/2)
# end: img_CAMERA1_1261230001.530205_right.jpg (index 800/2)
# For now two images from extract #07 (origin of coordinates are top left):
# img_CAMERA1_1261229993.980124_right.jpg (index 498/2)
#   - (399, 465) -- roundabout
#   - (619, 452) -- crossing
#   - (574, 449) -- give way
# img_CAMERA1_1261229994.980144_right.jpg (index 538/2)
#   - (375, 461) -- roundabout
#   - (600, 437) -- crossing
#   - (707, 426) -- give way

# TODO Check out the opencv function "solvePnP"

time_image_1 = 1261229993.980124
time_image_2 = 1261229994.980144

selector = np.logical_and(time_image_1-1 <= gps_timestamps, gps_timestamps <= time_image_2+1)
gps_times = gps_timestamps[selector]
gps_indices = np.where(selector)
print(gps_times)

# Values provided with as part of the dataset for Right Camera
resx = 1024
resy = 768
cx = 511.127987
cy = 388.337888
fx = 835.542079
fy = 837.180798
dist = [-3.508059e-001, 1.538358e-001, 0.000000e+000, 0.000000e+000, 0.000000e+000] # [K1 K2 T1 T2 K3]

# Camera matrix
K_right = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
    ])
K_left = np.array([
    [837.619011, 0, 522.434637],
    [0, 839.808333, 402.367400],
    [0, 0, 1],
    ])
# Distortion parameters
D = np.array(dist)
# TODO Find out what the "translation terms of the projection matrix" are and set them accordingly
Tx = 0
Ty = 0

translation_left_to_right = [1.194711e-001, -3.144088e-004, 1.423872e-004]
# Coordinate system information: https://en.wikipedia.org/wiki/Pinhole_camera_model
# Pixel coordinate will be rectified
# Camera coordinate system:
# x is right to camera
# y is up to camera
# z is away from camera
# Pixel coordinate system:
# Origin is bottom left
# x is to the right
# y is up
def project3dToPixel(xyz):
    u = (fx * xyz[0] + Tx) / xyz[2] + cx
    v = (fy * xyz[1] + Ty) / xyz[2] + cy
    return np.array([u, v])

def get3dRayFromPixel(uv,K):
    fx = K[0][0]
    fy=K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    ray_x = (uv[0] - cx)/fx
    ray_y = (uv[1] - cy)/fy
    return np.array([ray_x,ray_y])

def sample3dRayAt(rays,z,T=None):
    ray_x = rays[0]
    ray_y = rays[1]
    if T is None:
        return np.array([ray_x*z, ray_y*z, z])
    else:
        return np.array([ray_x*z + T[0], ray_y*z + T[1], z + T[2]]) 

# uv is assumed to be rectified
def projectPixelTo3dRay(uv, z=None):
    ray_x = (uv[0] - cx - Tx) / fx
    ray_y = (uv[1] - cy - Ty) / fy
    ray_z = 1.0;
    if z is not None:
        ray_z = z
    return np.array([ray_x, ray_y, ray_z])


img = mpimg.imread('data/images/img_CAMERA1_1261230069.530858_left.jpg')
px1 = project3dToPixel(np.array([0, 0, 10]))
px2 = project3dToPixel(np.array([0, -1, 10]))
px3 = project3dToPixel(np.array([0, -1, 30]))
print(px1)
print(px2)
print(px3)


uv_530858_left = [638,319]
uv_530858_right = [632,320]

zline = np.linspace(0, 5, 1000)
x_left,y_left,z_left = sample3dRayAt(get3dRayFromPixel(uv_530858_left,K_left),zline)
x_right,y_right,z_right = sample3dRayAt(get3dRayFromPixel(uv_530858_right,K_right),zline,translation_left_to_right)
# plt.imshow(img)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_left, y_left, z_left, 'gray')
ax.plot3D(x_right, y_right, z_right, 'blue')
# # ax = fig.add_subplot(111)

# # ax.scatter(px1[0], px1[1], color='r')
# # ax.scatter(px2[0], px2[1], color='g')
# # ax.scatter(px3[0], px3[1], color='b')

# # ax.set_xlim((0,resx))
# # ax.set_ylim((0,resy))

# #plt.axis('equal')
# plt.grid()

plt.show()
print(K_left[1][2])




















