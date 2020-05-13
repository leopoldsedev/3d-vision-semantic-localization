import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_lim_equal_scaling_3d(data_x, data_y, data_z):
    min_x = np.min(data_x)
    min_y = np.min(data_y)
    min_z = np.min(data_z)

    max_x = np.max(data_x)
    max_y = np.max(data_y)
    max_z = np.max(data_z)

    diff_x = np.abs(max_x - min_x)
    diff_y = np.abs(max_y - min_y)
    diff_z = np.abs(max_z - min_z)

    max_range = np.max([diff_x, diff_y, diff_z])

    mid_x = min_x + (diff_x) / 2.0
    mid_y = min_y + (diff_y) / 2.0
    mid_z = min_z + (diff_z) / 2.0

    return [(mid_x-max_range/2.0, mid_x+max_range/2.0), (mid_y-max_range/2.0, mid_y+max_range/2.0), (mid_z-max_range/2.0, mid_z+max_range/2.0)]


gps_full_data_6 = np.load('./transform_routes/transf_routes_overlap/route6_in_route7coords.npy')
gps_full_data_7 = np.genfromtxt('./07/gps.csv', skip_header=1)
gps_full_data_8 = np.load('./transform_routes/transf_routes_overlap/route8_in_route7coords.npy')
gps_full_data_10 = np.load('./transform_routes/transf_routes_overlap/route10_in_route7coords.npy')
gps_full_data_15 = np.load('./transform_routes/transf_routes_overlap/route15_in_route7coords.npy')

gps_local_6 = gps_full_data_6[:,8:11]
gps_local_7 = gps_full_data_7[:,8:11]
gps_local_8 = gps_full_data_8[:,8:11]
gps_local_10 = gps_full_data_10[:,8:11]
gps_local_15 = gps_full_data_15[:,8:11]

gps_x_6 = gps_local_6[:,0]
gps_y_6 = gps_local_6[:,1]
gps_z_6 = gps_local_6[:,2]

gps_x_7 = gps_local_7[:,0]
gps_y_7 = gps_local_7[:,1]
gps_z_7 = gps_local_7[:,2]

gps_x_8 = gps_local_8[:,0]
gps_y_8 = gps_local_8[:,1]
gps_z_8 = gps_local_8[:,2]

gps_x_10 = gps_local_10[:,0]
gps_y_10 = gps_local_10[:,1]
gps_z_10 = gps_local_10[:,2]

gps_x_15 = gps_local_15[:,0]
gps_y_15 = gps_local_15[:,1]
gps_z_15 = gps_local_15[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gps_x_6, gps_y_6, gps_z_6, s=2, color='blue')
ax.scatter(gps_x_7, gps_y_7, gps_z_7, s=2, color='red')
ax.scatter(gps_x_8, gps_y_8, gps_z_8, s=2, color='orange')
ax.scatter(gps_x_10, gps_y_10, gps_z_10, s=2, color='green')
ax.scatter(gps_x_15, gps_y_15, gps_z_15, s=2, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

limits = get_lim_equal_scaling_3d(np.hstack([gps_x_6, gps_x_7, gps_x_8, gps_x_10, gps_x_15]), np.hstack([gps_y_6, gps_y_7, gps_y_8, gps_y_10, gps_y_15]), np.hstack([gps_z_6, gps_z_7, gps_z_8, gps_z_10, gps_z_15]))
ax.set_xlim(limits[0])
ax.set_ylim(limits[1])
ax.set_zlim(limits[2])
plt.show()
