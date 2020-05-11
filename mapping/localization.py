import matplotlib.pyplot as plt
import numpy as np
import cv2
import transforms3d as tf

import prediction
import score
import detection
import mapping
import util
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera

MAP_PATH = './map.pickle'
QUERY_IMAGE_PATH = './07/images/rectified/img_CAMERA1_1261230001.030210_right.jpg'

def get_gaussian(sigma, mu):
    x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def get_pose_scores(landmark_list, query_image, camera, sign_types):
    query_detections, debug_image = detection.detect_traffic_signs_in_image(query_image, sign_types)
    #plt.imshow(util.bgr_to_rgb(debug_image))
    #plt.show()

    # TODO Use actual function
    #possible_camera_poses = get_possible_poses(landmark_list)
    size = 50
    possible_camera_poses = np.random.random((size, size, 8, 7))
    possible_camera_poses[:,:,:,2] = 0.0
    for x in range(possible_camera_poses.shape[0]):
        for y in range(possible_camera_poses.shape[1]):
            scale = 1
            possible_camera_poses[x,y,:,0] = x * scale
            possible_camera_poses[x,y,:,1] = y * scale

            for i in range(possible_camera_poses.shape[2]):
                yaw = np.random.random() * 360
                orientation = np.array(tf.euler.euler2quat(np.deg2rad(yaw), 0, 0, axes='szyx'))
                possible_camera_poses[x,y,i,3:7] = orientation

    def calculate_score(pose):
        # TODO Calculate and return actual score
        #predicted_detections = prediction.predicted_detections(pose, landmark_list, camera)
        #score = get_score(query_detections, predicted_detections, sign_types)
        #return score
        orientation = pose[3:7]
        yaw, _, _ = tf.euler.quat2euler(orientation, axes='szyx')
        if yaw < 0:
            yaw += 2 * np.pi
        return yaw

    scores = np.apply_along_axis(calculate_score, 3, possible_camera_poses)
    print(scores.shape)
    print(np.mean(np.max(scores, axis=2)))

    return possible_camera_poses, scores


def show_heatmap(possible_poses, pose_scores, landmark_list):
    print(pose_scores.shape)
    highest_score_per_position = np.max(pose_scores, axis=2)
    highest_score_pose_idx = np.argmax(pose_scores, axis=2)
    print(highest_score_per_position.shape)
    print(highest_score_pose_idx.shape)

    # Visualize as a heatmap
    #plt.imshow(highest_score_per_position, cmap='hot', interpolation='nearest')#, extent=[0, 200, -100, 100])
    plt.imshow(highest_score_per_position, cmap='viridis')#, extent=[0, 50, -100, 100])
    plt.colorbar()

    # Visualize orientations with highest score
    for x in range(possible_poses.shape[0]):
        for y in range(possible_poses.shape[1]):
            pose_idx = highest_score_pose_idx[x, y]
            pose = possible_poses[x, y, pose_idx, :]
            position = pose[0:3]
            orientation = pose[3:7]
            yaw, _, _ = tf.euler.quat2euler(orientation, axes='szyx')

            highest_arrow_length = 0.001
            dx = np.cos(yaw) * highest_arrow_length
            dy = np.sin(yaw) * highest_arrow_length
            plt.arrow(position[0], position[1], dx, dy, width = 0.1, color='black')
            #print(f'({position[0]},{position[1]})')


    for sign_type in sign_types:
        # Get landmarks from landmark list that are of same sign_type
        landmarks_of_type = list(filter(lambda l: l.sign_type == sign_type, landmark_list))

        # Extract landmark coordinates
        pos_x = list(map(lambda l: l.x, landmarks_of_type))
        pos_y = list(map(lambda l: l.y, landmarks_of_type))

        # Visualize landmark positions
        color = util.color_tuple_bgr_to_plt(detection.sign_type_colors[sign_type])
        plt.scatter(pos_x, pos_y, s=100, color=color)

        # Visualize landmark directions
        for landmark in landmarks_of_type:
            direction = landmark.direction
            # TODO Remove
            direction = np.array([-1.0, 0.0, 0.0])

            # Don't plot arrow if direction is all zeros
            if not direction.any():
                continue

            sign_arrow_length = 10
            dx = direction[0] * sign_arrow_length
            dy = direction[1] * sign_arrow_length
            plt.arrow(landmark.x, landmark.y, dx, dy, width = 3, color=color)

        print(list(landmarks_of_type))

    plt.show()


if __name__ == '__main__':
    landmark_list = mapping.load_map(MAP_PATH)
    query_image = cv2.imread(QUERY_IMAGE_PATH)
    assert(query_image is not None)

    camera = get_camera_malaga_extract_07_right()

    # TODO Use all sign types
    #sign_types = detection.ALL_SIGN_TYPES
    sign_types = {detection.TrafficSignType.CROSSING}
    sign_types = {}

    possible_poses, pose_scores = get_pose_scores(landmark_list, query_image, camera, sign_types)
    show_heatmap(possible_poses, pose_scores, landmark_list)

    # TODO Print best estimates
    # TODO Maybe refine pose around best estimates
