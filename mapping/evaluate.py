from sklearn.metrics import mean_squared_error as mse
import numpy as np
import random as rand
import localization
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera, malaga_car_pose_to_camera_pose
import detection
import util
import cv2
import sys
import pickle
from ground_truth_estimator import GroundTruthEstimator
np.set_printoptions(threshold=sys.maxsize)

# detections
timestamps = [1261230001.030210]
estimator = gt_estimator = GroundTruthEstimator('./data/07/gps.csv', './data/07/imu.csv', print_kf_progress=True)
positions = estimator.get_position(timestamps, method='cubic')
# orientations = estimator.get_pose(timestamps, method='cubic')
gps = positions[0][0:2]

# # integrate with localization
# MAP_PATH = './test/map_07.pickle'
# QUERY_IMAGE_PATH = '/home/patricia/3D/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images/img_CAMERA1_1261230001.030210_left.jpg'
# gps = np.array([151.53438214,-11.12351311])
# camera = get_camera_malaga_extract_07_right()
# sign_types = detection.ALL_SIGN_TYPES
# landmark_list = util.pickle_load(MAP_PATH)
# query_image = cv2.imread(QUERY_IMAGE_PATH)
# 
# # numbers of signs detected
# query_detections, debug_image = detection.detect_traffic_signs_in_image(query_image, sign_types)
# number_detections = np.asarray(query_detections).shape[0]
# print(number_detections)

# run localization before I get the pickle files of results
# possible_poses, pose_scores = localization.get_pose_scores(landmark_list, query_image, camera, sign_types)
# print(possible_poses.shape)
# print(pose_scores.shape)

# for dim in pose_scores.shape:
#     num_scores *= dim
# for i in range(num_scores):
#     pose_idx = np.unravel_index(i, pose_scores.shape)
#     scores.append(pose_scores[pose_idx])
#     poses.append(possible_poses[pose_idx])
with open('/home/patricia//Downloads/detections_07_right.pickle', 'rb') as file:
    detections = pickle.load(file)
with open('/home/patricia//Downloads/img_CAMERA1_1261230001.030210_right.jpg.pickle', 'rb') as file:
    pickle_scores = pickle.load(file)
with open('/home/patricia//Downloads/map_07_possible_poses.pickle', 'rb') as file:
    pickle_poses = pickle.load(file)
num_detections = np.asarray(detections.get('img_CAMERA1_1261230001.030210_right.jpg')).shape[0]
num_scores = 1
scores = []
poses = []

for dim in pickle_scores.shape:
    num_scores *= dim # a*b*c from dimension (a,b,c)
for i in range(num_scores):
    pose_idx = np.unravel_index(i, pickle_scores.shape)
    scores.append(pickle_scores[pose_idx])
    poses.append(pickle_poses[pose_idx])
scores = np.asarray([scores])
poses = np.asarray(poses)
print(scores.shape)
print(poses[:,0:2].shape)



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
sorted_combo_idx = np.argsort(combo[:,-1])
sorted_combo = combo[sorted_combo_idx]
sorted_predicts = np.delete(sorted_combo, np.s_[-1:], axis=1)
# leave only (x,y)
top_predicts = sorted_predicts[-9:]
print(top_predicts)

correct = 0 
error = []
for predict in top_predicts:
    error.append(mse(gps,predict))
    #print(predict,error)
print(error)
if np.min(error)<5:
    correct = 1
else:
    correct = 0
print(correct)
