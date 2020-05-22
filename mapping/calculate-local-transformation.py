import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
import sys

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

np.set_printoptions(threshold=100, linewidth=200)

#Get Matrix H_GL
print("-"*20)
print()
print()
print("Get Matrix H_GL and H_LG:")
print()
print()

gps_full_data = np.genfromtxt('MalagaDataSet_Routes/malaga-urban-dataset_GPS.txt', skip_header=1)

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
route_g1 = gps_geocen[543]
route_g2 = gps_geocen[544]
route_g3 = gps_geocen[545]
route_l1 = gps_local[543]
route_l2 = gps_local[544]
route_l3 = gps_local[545]

if np.linalg.matrix_rank(gps_geocen[1:4]) != 3 and np.linalg.matrix_rank(gps_local[1:4]) != 3:
    print("WARNING: Chosen vectors are not linearly independent")
    print("gps_geocen =", gps_geocen)
    print("gps_geocen.shape =", gps_geocen.shape)


# Build linear system of equations that constrains the rotation matrix
# TODO This could be done with all vectors in the list and the pseudo inverse would find the optimal fit
A = np.array([
    [route_l1[0], route_l1[1], route_l1[2], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, route_l1[0], route_l1[1], route_l1[2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, route_l1[0], route_l1[1], route_l1[2]],
    [route_l2[0], route_l2[1], route_l2[2], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, route_l2[0], route_l2[1], route_l2[2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, route_l2[0], route_l2[1], route_l2[2]],
    [route_l3[0], route_l3[1], route_l3[2], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, route_l3[0], route_l3[1], route_l3[2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, route_l3[0], route_l3[1], route_l3[2]],
    ])
b = np.array([
    route_g1[0] - t[0],
    route_g1[1] - t[1],
    route_g1[2] - t[2],
    route_g2[0] - t[0],
    route_g2[1] - t[1],
    route_g2[2] - t[2],
    route_g3[0] - t[0],
    route_g3[1] - t[1],
    route_g3[2] - t[2],
    ])

# Solve the system
print()
print("A.shape:", A.shape)
print("b.shape:", b.shape)
print()
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

print("Full 'local to geocen'-matrix\n")
print(H_GL)
print()
print("Full 'geocen to local'-matrix\n")
print(H_LG)
print()

#Print dot product to prove that the inverse was done correctly
print("Sanity check:")
print(np.dot(H_LG, H_GL))
print()

# Check result
for i in range(0, len(gps_local_y)):
    local_homogen = np.block([gps_local[i], 1])[np.newaxis].T
    geocen_transformed = np.dot(H_GL, local_homogen)[:-1]
    dist_geocen = np.linalg.norm(geocen_transformed - gps_geocen[i][np.newaxis].T)
    if not np.isclose(dist_geocen, 0):
        print("There is point that is not transformed right")

np.save('transform_routes/transf_matrices/Hfull_GL.npy',H_GL)

"""
#Check import
print("-"*20)
print()
print()
print("Check import")
print()
print()

H_LG_imported = np.load('transform_routes/transf_matrices/HX_LG.npy') #adapt
print(H_LG_imported)
"""