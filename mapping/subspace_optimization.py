import scipy
import numpy as np
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera
import prediction
from score import get_score, ALL_SIGN_TYPES
from localization import split_pose_array, POSITION_STEP_SIZE
from detection import TrafficSignType, TrafficSignDetection, ALL_SIGN_TYPES
import transforms3d as tf

# optimization

def convert_ImagePose_to_np_array(pose):
    return np.concatenate((pose.position, pose.orientation), axis=None)

def calculate_guassian_score(pose, landmark_list, query_detections, camera, sign_types):
    position, orientation = split_pose_array(pose)
    image_pose = ImagePose(position=position, orientation=orientation)

    debug = False
    dist = np.linalg.norm(np.array([150.0, -6.0]) - position[0:2])
    #if dist < 2:
        #debug = True

    predicted_detections = prediction.predicted_detections(image_pose, landmark_list, camera, debug=debug)

    _, _, yaw = tf.euler.quat2euler(orientation, axes='sxyz')

    # If there are no detections predicted use the cached score for that case
    if len(predicted_detections) == 0:
        score_val = empty_predicted_score
    else:
        score_val = get_score(predicted_detections, query_detections, sign_types, debug=debug)
    
    return score_val

def calc_score(x, query_detection,landmark_list, camera, roll, pitch, sign_types):
    pos = x[0:3]
    yaw = x[3]
    orientation = tf.euler.euler2quat(roll, pitch, yaw, 'sxyz')
    x = np.concatenate((pos,orientation))
    score = calculate_guassian_score(x, landmark_list, query_detections, camera, sign_types)
    return 1 - score

# TODO: Think of a better name for this function
# INPUTS: initial_pose - IMAGE_POSE of the initial pose
#         query_detections - detections in query image
#         landmark_list - list of landmarks
#          camera - camera parameters
# Output: new position array after optimization


# consider the yaw as an optimization variable and fix the other 2
def optimize_over_space(initial_pose,query_detections, landmark_list, camera, sign_types=ALL_SIGN_TYPES):
    
    initial_position = initial_pose.position
    initial_orientation = initial_pose.orientation
    roll, pitch, yaw = tf.euler.quat2euler(initial_orientation, axes='sxyz')

    def constraint(x):
        assert(len(x) == 4)
        if (x[0] > initial_position[0] + POSITION_STEP_SIZE/2) or (x[0] < initial_position[0] - POSITION_STEP_SIZE/2):
            return -1
        elif (x[1] > initial_position[1] + POSITION_STEP_SIZE/2) or (x[1] < initial_position[1] - POSITION_STEP_SIZE/2):
            return -1
        else:
            return 0
    cons = [{'type':'ineq', 'fun': constraint}]
    initial_input = np.concatenate((initial_position,np.atleast_1d(yaw)))
    output = scipy.optimize.minimize(calc_score, initial_input, args=(query_detections, landmark_list, camera, roll, pitch, sign_types), constraints=cons)
    print(output)
    final_pose = output.x[0:3]
    final_yaw = output.x[3]
    final_orientation = tf.euler.euler2quat(roll, pitch, final_yaw, 'sxyz')
    ret = ImagePose(position=final_pose,orientation=final_orientation)
    return ret

if __name__ == '__main__':
    landmark1 = MapLandmark(x=800, y=400, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark2 = MapLandmark(x=900, y=500, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark3 = MapLandmark(x=1, y=3, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark4 = MapLandmark(x=1, y=3, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark5 = MapLandmark(x=0, y=4, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark6 = MapLandmark(x=0, y=4, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark7 = MapLandmark(x=1, y=4, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark8 = MapLandmark(x=1, y=4, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark_list = [landmark1, landmark2, landmark3, landmark4, landmark5, landmark6, landmark7, landmark8]
    detection1 = TrafficSignDetection(x=809.0, y=408.0, width=38, height=38, sign_type=TrafficSignType.CROSSING, score=0.8859539)
    detection2 = TrafficSignDetection(x=205.0, y=428.0, width=30, height=30, sign_type=TrafficSignType.CROSSING, score=0.89329803)
    query_detections = [detection1, detection2]
    camera = get_camera_malaga_extract_07_right()

    cam_pos = np.array([0, 0, 0])
    yaw_deg = -90
    yaw_deg = 0
    cam_rot = tf.euler.euler2mat(np.deg2rad(-90), 0, np.deg2rad(yaw_deg), 'sxyz')
    pose = ImagePose(orientation=tf.quaternions.mat2quat(cam_rot), position=cam_pos)
    print("input - {}".format(pose))
    ret = optimize_over_space(pose,query_detections,landmark_list,camera, ALL_SIGN_TYPES)
    print("output - {}".format(ret))