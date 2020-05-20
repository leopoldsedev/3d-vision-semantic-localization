from sklearn.metrics import mean_squared_error as mse
import numpy as np
import random as rand
import localization
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera, malaga_car_pose_to_camera_pose
import detection
import util
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)
# integrate with localization
MAP_PATH = './test/map_07.pickle'
QUERY_IMAGE_PATH = '/home/patricia/3D/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images/img_CAMERA1_1261230001.030210_left.jpg'
gps = np.array([151.53438214,-11.12351311])
camera = get_camera_malaga_extract_07_right()
sign_types = detection.ALL_SIGN_TYPES
landmark_list = util.pickle_load(MAP_PATH)
query_image = cv2.imread(QUERY_IMAGE_PATH)

# # numbers of signs detected
# query_detections, debug_image = detection.detect_traffic_signs_in_image(query_image, sign_types)
# number_detections = np.asarray(query_detections).shape[0]
# print(number_detections)

possible_poses, pose_scores = localization.get_pose_scores(landmark_list, query_image, camera, sign_types)
print(possible_poses.shape)
print(pose_scores.shape)

# experiment 
num_scores = 1
scores = []
poses = []
for dim in pose_scores.shape:
    num_scores *= dim
for i in range(num_scores):
    pose_idx = np.unravel_index(i, pose_scores.shape)
    scores.append(pose_scores[pose_idx])
    poses.append(possible_poses[pose_idx])
scores = np.asarray([scores])
poses = np.asarray(poses)
print(scores.shape)
print(poses[:,0:2].shape)


# take away dimension: ang
possible_poses = np.asarray(possible_poses[:,:,0,:])

# print(pose_scores)
pose_scores = np.asarray(pose_scores[:,:,0])
# print(possible_poses,pose_scores)
estimates = []
errors = []
for x in possible_poses:
    for i in range(possible_poses.shape[1]):
        #print(x[0][0],x[i,1])
        estimates.append((x[0][0],x[i,1]))
for e in pose_scores:
    #print("new"+str(e))
    for i in range(pose_scores.shape[1]):
        errors.append(e[i])
estimates = np.asarray(estimates)
errors = np.asarray([errors])
# x_min = np.min(estimates[:,0])
# x_max = np.max(estimates[:,0])
# y_min = np.min(estimates[:,1])
# y_max = np.max(estimates[:,1])
# print(x_min,x_max,y_min,y_max)
# print(estimates)
# print(errors)
# print(errors.shape)
# print(estimates.shape)
# print(errors.shape)


# # before we can read the text files..
# position = np.array([-3.60000000e+01, -4.70000000e+01 , 0,  7.55853469e-01 , 2,     -6.54740814e-01,  0.00000000e+00,  0.00000000e+00])
# true_position = np.array([1,1])
# # create predictions w/ rand noise and scores
# predicts = [0,0,40]
# scores = [0]
# for i in range(10):
#     predict = np.array([i+rand.uniform(0,1),i+rand.uniform(0,1),40])
#     predicts = np.vstack((predicts, predict))
#     score = np.array([-i+rand.uniform(0,1)])
#     scores = np.vstack((scores,score))
# predicts = np.asarray(predicts)
# 
# 
# # predicts = (x,y,ang), needs to strip away ang
# predicts = np.delete(predicts, np.s_[-1:], axis=1)

combo = np.append(poses[:,0:2],scores.T, axis=1)
print(combo)
sorted_combo_idx = np.argsort(combo[:,-1])
sorted_combo = combo[sorted_combo_idx]
# print(sorted_combo)
sorted_predicts = np.delete(sorted_combo, np.s_[-1:], axis=1)
# leave only (x,y)
top_predicts = sorted_predicts[-9:]
for predict in top_predicts:
    error = mse(gps,predict)
    print(predict,error)
