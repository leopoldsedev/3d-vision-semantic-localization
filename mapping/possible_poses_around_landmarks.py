import numpy as np
import triangulation as tr

# Input: landmark list (as landmark_list -> output of mapping.py)
landmark1 = tr.MapLandmark('MapLandmark', [10, 10, 0, 'CROSSING', 0.5, [0.0, 0.0, 0.0]])
landmark2 = tr.MapLandmark('MapLandmark', [-50, -100, 0, 'CROSSING', 0.9, [0.0, 0.0, 0.0]])
landmark3 = tr.MapLandmark('MapLandmark', [100, -500, 0, 'YIELD', 0.94, [0.0, 0.0, 0.0]])

print(landmark1)

thresh_side = 10
step_size = 1
num_poses = thresh_side**2 / step_size

potential_poses = np.zeros([num_poses, 6])

for landmark in landmark_names:
    for pose in range(potential_poses):
        #Define thresholds in both axis for the position mesh
        landmark_x_low = landmark.x-thresh_side
        landmark_x_high = landmark.x+thresh_side
        landmark_y_low = landmark.y+thresh_side
        landmark_y_high = landmark.y+thresh_side

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