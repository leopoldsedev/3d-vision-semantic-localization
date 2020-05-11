import numpy as np
import pandas as pd
import triangulation as tr
import transforms3d as tf

# Input: landmark list (as landmark_list -> output of mapping.py)
landmark1 = tr.MapLandmark(x=139.4, y=75.4, z=0, sign_type='CROSSING', confidence_score=0.5, direction=[0.0, 0.0, 0.0])
landmark2 = tr.MapLandmark(x=506, y=1184.9, z=0, sign_type='CROSSING', confidence_score=0.5, direction=[0.0, 0.0, 0.0])
landmark3 = tr.MapLandmark(x=124.7, y=62.3, z=0, sign_type='CROSSING', confidence_score=0.5, direction=[0.0, 0.0, 0.0])

landmark_names = [landmark1, landmark2, landmark3]

#Define map_mesh
    #look at min max landmarks gpa coordinates and add threshhold
    #import gps coordinates
    #coord_df = open(r"MalagaDataSet_Route7/malaga-urban-dataset-extract-07_all-sensors_GPS.txt")
"""coor_df = pd.read_csv('MalagaDataSet_Route7/malaga-urban-dataset-extract-07_all-sensors_GPS.txt', sep="                 ", header=None)
print("coor_df.shape:\n", coor_df.shape)
print()
print("coor_df.head:\n", coor_df.head())"""
    #Defined dummy since problems importing the GPS from the txt file
min_x, max_x = 100, 1200
min_y, max_y = 50, 1500

thresh_side = 10
step_size = 5

x_values_num = int((max_x-min_x)/step_size)
y_values_num = int((max_y-min_y)/step_size)
total_values_num = x_values_num * y_values_num

#Create map_mesh that covers the whole map, where columns=['x coord', 'y coord', '# signals around']
map_mesh = np.zeros([total_values_num, 3])
for x in range (x_values_num):
    for y in range(y_values_num):
        map_mesh[y_values_num*x+y,0] = min_x + x * step_size       
        map_mesh[y_values_num*x+y,1] = min_y + y * step_size
        map_mesh[y_values_num*x+y,2] = 0

#Calculate points in the map_mesh that are close to the landmark
for landmark in landmark_names:
    landmark_x_low = landmark.x - thresh_side
    landmark_x_high = landmark.x + thresh_side
    landmark_y_low = landmark.y - thresh_side
    landmark_y_high = landmark.y + thresh_side
    
    print()
    print("----------")
    print("----------")
    print()
    print("Landmark:", landmark)
    print("Landmark search grid:", "(", landmark_x_low, "-", landmark_x_high, ")", " , ", "(", landmark_y_low, "-", landmark_y_high, ")")

    #Create a mesh around landmark of potential points.
    potential_pos_mesh = np.zeros([1,2])

    for row in map_mesh:
        if (row[0] >=  landmark_x_low) and (row[0] <= landmark_x_high) and (row[1] >= landmark_y_low) and (row[1] <= landmark_y_high):
            conc_row = np.zeros([1,2])
            conc_row[0,0] = row[0]
            conc_row[0,1] = row[1]
            potential_pos_mesh = np.concatenate((potential_pos_mesh,conc_row))

    #delete first row of potential_pos_mesh since only used to initialize array
    potential_pos_mesh = potential_pos_mesh[1:, :]
    
    print()
    print("potential_pos_mesh:\n", potential_pos_mesh)
    print()

    #Sum 1 to the third column of the mesh-map in the coordinates that are close to a landmark
    for map_row in map_mesh:
        for pos_row in potential_pos_mesh:
            if (map_row[0]==pos_row[0]) and (map_row[1]==pos_row[1]):
                map_row[2] = map_row[2]+1
 
print("----------")
print("----------")
print("----------")
print()

print("TEST")
print("map_mesh:\n", map_mesh[1730:1750,:])
print()
print("map_mesh:\n", map_mesh[23710:23730,:])
print()
print("----------")
print("----------")
print("----------")
print()


#Christians Matrix 
print("Christians matrix")

angle_step_size = 60 #in degrees
num_angles = 360 / angle_step_size
# Check if angle step size divides into whole numbers
assert(np.allclose(num_angles, np.floor(num_angles)))
num_angles = int(num_angles)

landmark_x_low = np.min([landmark.x for landmark in landmark_names])
landmark_y_low = np.min([landmark.y for landmark in landmark_names])
landmark_x_high = np.max([landmark.x for landmark in landmark_names])
landmark_y_high = np.max([landmark.y for landmark in landmark_names])

x_low = landmark_x_low - thresh_side
y_low = landmark_y_low - thresh_side
x_high = landmark_x_high + thresh_side
y_high = landmark_y_high + thresh_side

x_steps = range(int(np.floor(x_low)), int(np.ceil(x_high)) + step_size, step_size)
y_steps = range(int(np.floor(y_low)), int(np.ceil(y_high)) + step_size, step_size)
angle_steps = range(0, 360, angle_step_size)

possible_poses = np.zeros((len(x_steps), len(y_steps), len(angle_steps), 7))

for i, x in enumerate(x_steps):
    for j, y in enumerate(y_steps):
        for k, yaw in enumerate(angle_steps):
            position = np.array([x, y, 0.0])
            orientation = np.array(tf.euler.euler2quat(np.deg2rad(yaw), 0, 0, axes='szyx'))
            possible_poses[i, j, k] = np.hstack((position, orientation))

"""
#orientation has to be a quaternion
potential_poses = np.zeros([num_poses, 6])

for landmark in landmark_names:
    for pose in range(potential_poses):
        #Define thresholds in both axis for the position mesh
        landmark_x_low = landmark.x - thresh_side
        landmark_x_high = landmark.x + thresh_side
        landmark_y_low = landmark.y - thresh_side
        landmark_y_high = landmark.y + thresh_side

        #Iterate over potential positions
        for x_pose in range(landmark_x_low, landmark_x_high, step_size):
            for y_pose in range(landmark_y_low, landmark_y_high, step_size):
                potential_poses[pose,0] = x_pose
                potential_poses[pose,1] = y_pose
                potential_poses[pose,2] = 0
                potential_poses[pose,3] = landmark.x - x_pose
                potential_poses[pose,4] = landmark.y - y_pose
                potential_poses[pose,5] = landmark.z

            #normalize
                potential_poses[pose,3] = potential_poses[pose,3] / (potential_poses[pose,3]**2 + potential_poses[pose,4]**2+ potential_poses[pose,5]**2)**0.5
                potential_poses[pose,4] = potential_poses[pose,4] / (potential_poses[pose,3]**2 + potential_poses[pose,4]**2+ potential_poses[pose,5]**2)**0.5
                potential_poses[pose,5] = potential_poses[pose,5] / (potential_poses[pose,3]**2 + potential_poses[pose,4]**2+ potential_poses[pose,5]**2)**0.5
"""
