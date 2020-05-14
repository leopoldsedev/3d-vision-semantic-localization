import scipy
import numpy as np
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera
from prediction import predicted_detections
from ALL_SIGN_TYPES, score import get_score

def convert_ImagePose_to_np_array(pose):
    return np.concatenate((pose.position, pose.orientation), axis=None)

def convert_np_array_to_ImagePose(pose):
    return ImagePose(orientation=pos[3:7], position=pose[0:3])

def calc_score(x, query_detection,landmark_list, camera):
    pose = convert_np_array_to_ImagePose(x)
    detections_predicted = predicted_detections(pose, landmark_list, camera)
    score = get_score(query_detection, detections_predicted, ALL_SIGN_TYPES)
    return score


def optimize_over_space(query_detection,initial_pose, landmark_list, camera):
    temp = np.concatenate((pose.position, pose.orientation), axis=None)
    pose = convert_ImagePose_to_np_array(initial_pose)
    ret = scipy.optimize.minimize(get_score, initial_pose, args=(query_detection, landmark_list, camera))
    return ret