import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
import sys

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

np.set_printoptions(threshold=100, linewidth=200)


gps_full_data = np.genfromtxt('07/gps.csv', skip_header=1)

gps_timestamps = gps_full_data[:,0]
gps_geocen_x = gps_full_data[:,12]
gps_geocen_y = gps_full_data[:,13]
gps_geocen_z = gps_full_data[:,14]
gps_geocen = np.block([[gps_geocen_x], [gps_geocen_y], [gps_geocen_z]]).T # size n x 3
gps_local_x = gps_full_data[:,8]
gps_local_y = gps_full_data[:,9]
gps_local_z = gps_full_data[:,10]
gps_local = np.block([[gps_local_x], [gps_local_y], [gps_local_z]]).T # size n x 3


for i in range(1, len(gps_local_x)):
    dist_local = np.linalg.norm(gps_local[i-1] - gps_local[i])
    dist_geocen = np.linalg.norm(gps_geocen[i-1] - gps_geocen[i])
    if np.isclose(dist_local, dist_geocen):
        #print("{} is equal".format(i))
        pass
    else:
        print("Mismatch at {}: {} != {}".format(i, dist_local, dist_geocen))
        break

# The first point from the geocenter points is the translational part, because the first local point is the local origin
t = np.array(gps_geocen[0])

# Choose 3 linearly independent points
g1 = gps_geocen[1]
g2 = gps_geocen[2]
g3 = gps_geocen[3]
l1 = gps_local[1]
l2 = gps_local[2]
l3 = gps_local[3]

if np.linalg.matrix_rank(gps_geocen[1:4]) != 3 and np.linalg.matrix_rank(gps_local[1:4]) != 3:
    print("WARNING: Chosen vectors are not linearly independent")

# Build linear system of equations that constrains the rotation matrix
# TODO This could be done with all vectors in the list and the pseudo inverse would find the optimal fit
A = np.array([
    [l1[0], l1[1], l1[2], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, l1[0], l1[1], l1[2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, l1[0], l1[1], l1[2]],
    [l2[0], l2[1], l2[2], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, l2[0], l2[1], l2[2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, l2[0], l2[1], l2[2]],
    [l3[0], l3[1], l3[2], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, l3[0], l3[1], l3[2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, l3[0], l3[1], l3[2]],
    ])
b = np.array([
    g1[0] - t[0],
    g1[1] - t[1],
    g1[2] - t[2],
    g2[0] - t[0],
    g2[1] - t[1],
    g2[2] - t[2],
    g3[0] - t[0],
    g3[1] - t[1],
    g3[2] - t[2],
    ])

# Solve the system
r = np.linalg.solve(A, b)

# Build rotation matrix
R = np.array([
    [r[0], r[1], r[2]],
    [r[3], r[4], r[5]],
    [r[6], r[7], r[8]],
    ])

# Build complete transformation from local frame to geocen frame
H_GL = tf.affines.compose(t, R, np.ones(3), np.zeros(3))
# Calculate the inverse
H_LG = tf.affines.compose(np.dot(-R.T, t), R.T, np.ones(3), np.zeros(3))

print(H_GL)
print(H_LG)
print(np.dot(H_LG, H_GL))

# Check result
for i in range(0, len(gps_local_x)):
    local_homogen = np.block([gps_local[i], 1])[np.newaxis].T
    geocen_transformed = np.dot(H_GL, local_homogen)[:-1]
    dist_geocen = np.linalg.norm(geocen_transformed - gps_geocen[i][np.newaxis].T)
    if not np.isclose(dist_geocen, 0):
        print("There is point that is not transformed right")
