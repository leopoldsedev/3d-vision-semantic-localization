import matplotlib.pyplot as plt
import numpy as np
import cv2
import transforms3d as tf
from numpy import interp

import prediction
import score
import detection
import mapping
import util
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera, malaga_car_pose_to_camera_pose

MAP_PATH = './map.pickle'
QUERY_IMAGE_PATH = './07/images/rectified/img_CAMERA1_1261230001.030210_right.jpg'
#QUERY_IMAGE_PATH = './07/images/rectified/img_CAMERA1_1261229994.680132_right.jpg'

POSITION_STEP_SIZE = 5
ANGLE_STEP_SIZE = 10
LANDMARK_MARGIN = prediction.LANDMARK_RELEVANCE_RANGE


def get_possible_poses(landmark_list, position_step_size, angle_step_size, landmark_margin):
    num_angles = 360 / angle_step_size
    # Check if angle step size divides into whole numbers
    assert(np.allclose(num_angles, np.floor(num_angles)))

    landmark_x_low = np.min([landmark.x for landmark in landmark_list])
    landmark_y_low = np.min([landmark.y for landmark in landmark_list])
    landmark_x_high = np.max([landmark.x for landmark in landmark_list])
    landmark_y_high = np.max([landmark.y for landmark in landmark_list])

    x_low = landmark_x_low - landmark_margin
    y_low = landmark_y_low - landmark_margin
    x_high = landmark_x_high + landmark_margin
    y_high = landmark_y_high + landmark_margin

    x_steps = range(int(np.floor(x_low)), int(np.ceil(x_high)) + position_step_size, position_step_size)
    y_steps = range(int(np.floor(y_low)), int(np.ceil(y_high)) + position_step_size, position_step_size)
    angle_steps = range(0, 360, angle_step_size)

    possible_poses = np.zeros((len(x_steps), len(y_steps), len(angle_steps), 7))

    for i, x in enumerate(x_steps):
        for j, y in enumerate(y_steps):
            for k, yaw_deg in enumerate(angle_steps):
                position_car = np.array([x, y, 0.0])
                orientation_car = tf.euler.euler2quat(np.deg2rad(-90), 0, np.deg2rad(yaw_deg), 'sxyz')

                position_camera, orientation_camera = malaga_car_pose_to_camera_pose(position_car, orientation_car, right=True)

                possible_poses[i, j, k] = np.hstack((position_camera, orientation_camera))

    return possible_poses


score_calc_count = 0
def get_pose_scores(landmark_list, query_detections, possible_camera_poses, camera, sign_types):
    # Filter only landmarks of given sign types
    landmark_list = list(filter(lambda l: l.sign_type in sign_types, landmark_list))

    empty_predicted_score = score.get_score([], query_detections, sign_types, debug=False)

    total_poses = possible_camera_poses.size / 7
    score_calc_count = 0

    def calculate_score(pose):
        position, orientation = split_pose_array(pose)
        image_pose = ImagePose(position=position, orientation=orientation)

        debug = False
        dist = np.linalg.norm(np.array([150.0, -6.0]) - position[0:2])
        #if dist < 2:
            #debug = True

        predicted_detections = prediction.predicted_detections(image_pose, landmark_list, camera, debug=debug)

        _, _, yaw = tf.euler.quat2euler(orientation, axes='sxyz')

        if debug:
            debug_image = np.zeros((camera.height, camera.width, 3))
            debug_image = detection.generate_debug_image(debug_image, predicted_detections)
            plt.imshow(util.bgr_to_rgb(debug_image)/255)
            plt.show()

        # If there are no detections predicted use the cached score for that case
        if len(predicted_detections) == 0:
            score_val = empty_predicted_score
        else:
            score_val = score.get_score(predicted_detections, query_detections, sign_types, debug=debug)

        global score_calc_count
        score_calc_count += 1
        progress = 100 * score_calc_count / total_poses
        print('{:.2f}%    '.format(progress), end='\r')
        return score_val

    scores = np.apply_along_axis(calculate_score, 3, possible_camera_poses)

    print('Done     ')

    return possible_camera_poses, scores


def split_pose_array(pose_arr):
    assert(pose_arr.shape == (7,))
    position = pose_arr[0:3]
    orientation = pose_arr[3:7]
    return position, orientation


def show_heatmap(possible_poses, pose_scores, landmark_list):
    highest_score_per_position = np.max(pose_scores, axis=2)
    highest_score_pose_idx = np.argmax(pose_scores, axis=2)

    extent = [
        np.min(possible_poses[:,0,0,0]) - POSITION_STEP_SIZE/2,
        np.max(possible_poses[:,0,0,0]) + POSITION_STEP_SIZE/2,
        # The y-axis is flipped intentionally (because in image coordinates the y-axis starts from the top-left corner)
        np.max(possible_poses[0,:,0,1]) + POSITION_STEP_SIZE/2,
        np.min(possible_poses[0,:,0,1]) - POSITION_STEP_SIZE/2,
    ]

    # Adjust scores to look nicer in visualization
    # TODO Maybe don't do this if query image has no detections
    empty_predicted_score = score.get_score([], [], sign_types, debug=False)
    highest_scores_adjusted = interp(highest_score_per_position, [empty_predicted_score, 1], [0, 1])
    heatmap_values = highest_scores_adjusted.T

    # Visualize as a heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.imshow(heatmap_values.T, cmap='hot', interpolation='nearest', extent=[0, 200, -100, 100])
    #ax.imshow(heatmap_values.T, cmap='viridis', extent=extent)
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    im = ax.imshow(heatmap_values, cmap='Greens', extent=extent)
    #im.set_clim(0.0, 1.0)
    fig.colorbar(im, cax=cax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.set_xlim((extent[0], extent[1]))
    # Note that the y limits were flipped in extent
    ax.set_ylim((extent[3], extent[2]))

    poses = possible_poses[:,:,0,0:2]
    poses = np.reshape(poses, (poses.shape[0]*poses.shape[1], 2))
    ax.scatter(poses[:,0], poses[:,1], s=0.1, color='black')

    grid_element_size = (extent[1] - extent[0]) / heatmap_values.shape[1]
    arrow_width = grid_element_size / 20

    # Highlight best score with red circle
    radius = np.linalg.norm(grid_element_size / 2 * np.ones(2))
    best_pose_idx = np.unravel_index(np.argmax(pose_scores), pose_scores.shape)
    best_pose = possible_poses[best_pose_idx]
    best_pose_position = (best_pose[0], best_pose[1])
    best_pose_circle = plt.Circle(best_pose_position, radius, color='red', fill=False)
    ax.add_artist(best_pose_circle)

    # Visualize orientations with highest score at each position
    for x in range(possible_poses.shape[0]):
        for y in range(possible_poses.shape[1]):
            # Don't plot arrow if score is zero
            if np.allclose(heatmap_values.T[x, y], 0.0):
                continue

            pose_idx = highest_score_pose_idx[x, y]
            pose = possible_poses[x, y, pose_idx, :]
            position, orientation = split_pose_array(pose)
            _, _, yaw = tf.euler.quat2euler(orientation, axes='sxyz')

            highest_arrow_length = 0.001
            # Yaw needs to be offset by +90 degrees to be the correct direction in the visualization
            dx = np.cos(yaw + np.deg2rad(90)) * highest_arrow_length
            dy = np.sin(yaw + np.deg2rad(90)) * highest_arrow_length
            ax.arrow(position[0], position[1], dx, dy, width = arrow_width, color='black')


    for sign_type in sign_types:
        # Get landmarks from landmark list that are of same sign_type
        landmarks_of_type = list(filter(lambda l: l.sign_type == sign_type, landmark_list))

        # Extract landmark coordinates
        pos_x = list(map(lambda l: l.x, landmarks_of_type))
        pos_y = list(map(lambda l: l.y, landmarks_of_type))

        # Visualize landmark positions
        color = util.color_tuple_bgr_to_plt(detection.sign_type_colors[sign_type])
        ax.scatter(pos_x, pos_y, s=100, color=color)

        # Visualize landmark directions
        for landmark in landmarks_of_type:
            direction = landmark.direction

            # Don't plot arrow if direction is all zeros
            if not direction.any():
                continue

            sign_arrow_length = 1
            dx = direction[0] * sign_arrow_length
            dy = direction[1] * sign_arrow_length
            ax.arrow(landmark.x, landmark.y, dx, dy, width = 2*arrow_width, color=color)

    plt.show()


if __name__ == '__main__':
    landmark_list = util.pickle_load(MAP_PATH)
    query_image = cv2.imread(QUERY_IMAGE_PATH)
    assert(query_image is not None)

    camera = get_camera_malaga_extract_07_right()

    sign_types = detection.ALL_SIGN_TYPES

    query_detections, debug_image = detection.detect_traffic_signs_in_image(query_image, sign_types)
    # TODO Uncomment
#     detection1 = detection.TrafficSignDetection(x=809.0, y=408.0, width=38, height=38, sign_type=detection.TrafficSignType.CROSSING, score=0.8859539)
#     detection2 = detection.TrafficSignDetection(x=205.0, y=428.0, width=30, height=30, sign_type=detection.TrafficSignType.CROSSING, score=0.89329803)
#     query_detections = [detection1, detection2]
    plt.imshow(util.bgr_to_rgb(debug_image))
    plt.show(block=False)
    # Pause so that the window gets drawn
    plt.pause(0.0001)

    possible_camera_poses = get_possible_poses(landmark_list, POSITION_STEP_SIZE, ANGLE_STEP_SIZE, LANDMARK_MARGIN)

    possible_poses, pose_scores = get_pose_scores(landmark_list, query_detections, possible_camera_poses, camera, sign_types)
    show_heatmap(possible_poses, pose_scores, landmark_list)

    # TODO Print best estimates
    # TODO Maybe refine pose around best estimates
