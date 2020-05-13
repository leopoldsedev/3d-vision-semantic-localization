import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
import sys

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

np.set_printoptions(threshold=100, linewidth=200)

# INSTRUCTIONS
#Change lines 22, 24 and 57 to match the route you want to converto to local7 coords

#Get Matrix H_GL
print("-"*20)
print()
print()
print("Transform local coordinates of route to local coordinates of route 7:")
print()
print()

gps_full_data = np.genfromtxt('MalagaDataSet_Routes/malaga-urban-dataset-extract-[number]_all-sensors_GPS.txt', skip_header=1) #malaga-urban-dataset-extract-[number]_all-sensors_GPS.txt
H7_LG = np.load("transform_routes/transf_matrices/H7_LG.npy")
H_GL = np.load("transform_routes/transf_matrices/H[number]_GL.npy") #(H[number]_GL)")


gps_timestamps = gps_full_data[:,0]
print("gps_full_data.shape BEFORE:", gps_full_data.shape)
print("gps_full_data.shape head():", gps_full_data[0:5,6:13])
gps_local_x = gps_full_data[:,8]
gps_local_y = gps_full_data[:,9]
gps_local_z = gps_full_data[:,10]
gps_local = np.block([[gps_local_x], [gps_local_y], [gps_local_z]]).T # size n x 3

#Transform localX to geocentered
local_transformed = np.zeros([1,3])

for i in range(0, len(gps_local_x)):
    local_homogen = np.block([gps_local[i], 1])[np.newaxis].T
    geocen_transformed_line = np.dot(H7_LG, local_homogen)
    local_transformed_line = np.dot(H_GL, geocen_transformed_line)[:-1] #delete the final 1 before concatenating
    #Overwrite gps_full_data local7 coordinates
    gps_full_data[i,8] = local_transformed_line[0]
    gps_full_data[i,9] = local_transformed_line[1]
    gps_full_data[i,10] = local_transformed_line[2]
    #append to full matrix
    local_transformed = np.concatenate([local_transformed, local_transformed_line.T])

#Delete first row since only for initialisation
local_transformed = local_transformed[1:,:]

print()
print("gps_full_data.shape AFTER:", gps_full_data.shape)
print("gps_full_data.head AFTER:", gps_full_data[0:5,6:13])
print()

np.save("transform_routes/transf_routes_coords/route[number]_in_route7coords", gps_full_data) #route[number]_in_route7coords