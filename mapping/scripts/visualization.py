import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import cv2
import transforms3d as tf
from numpy import interp

import util
import detection
import localization
import score
import prediction
from ground_truth_estimator import GroundTruthEstimator, load_gps_and_imu_data
from triangulation import get_camera_malaga_extract_07_right, ImagePose

POSITION_STEP_SIZE = 5
ANGLE_STEP_SIZE = 10


def split_pose_array(pose_arr):
    assert(pose_arr.shape == (7,))
    position = pose_arr[0:3]
    orientation = pose_arr[3:7]
    return position, orientation


def visualize_landmarks(ax, landmark_list, sign_types, landmark_arrow_width):
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
            ax.arrow(landmark.x, landmark.y, dx, dy, width = landmark_arrow_width, color=color)


def show_heatmap(possible_poses, pose_scores, landmark_list, sign_types):
    highest_score_per_position = np.max(pose_scores, axis=2)
    highest_score_pose_idx = np.argmax(pose_scores, axis=2)

    extent = [
        np.min(possible_poses[:,0,0,0]) - POSITION_STEP_SIZE/2,
        np.max(possible_poses[:,0,0,0]) + POSITION_STEP_SIZE/2,
        # The y-axis is flipped intentionally (because in image coordinates the y-axis starts from the top-left corner)
        np.max(possible_poses[0,:,0,1]) + POSITION_STEP_SIZE/2,
        np.min(possible_poses[0,:,0,1]) - POSITION_STEP_SIZE/2,
    ]
    print(extent)

    # Adjust scores to look nicer in visualization
    # TODO Maybe don't do this if query image has no detections
    empty_predicted_score = score.get_score([], [], sign_types, debug=False)
    highest_scores_adjusted = interp(highest_score_per_position, [empty_predicted_score, 1], [0, 1])
    heatmap_values = highest_scores_adjusted.T
#     heatmap_values = np.zeros(heatmap_values.shape)

    # Visualize as a heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    im = ax.imshow(heatmap_values, cmap='Greens', extent=extent)
    fig.colorbar(im, cax=cax)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # Note that the y limits were flipped in extent
    ax.axis([extent[0], extent[1], extent[3], extent[2]], aspect='equal')

    poses = possible_poses[:,:,0,0:2]
    poses = np.reshape(poses, (poses.shape[0]*poses.shape[1], 2))
    ax.scatter(poses[:,0], poses[:,1], s=0.1, color='black')

    grid_element_size = (extent[1] - extent[0]) / heatmap_values.shape[1]
    arrow_width = grid_element_size / 20

    # Highlight best score with red circle
    radius = np.linalg.norm(grid_element_size / 2 * np.ones(2))
    best_pose_idx = np.unravel_index(np.argmax(pose_scores), pose_scores.shape)
    best_pose = possible_poses[best_pose_idx]
    print(best_pose)
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

    visualize_landmarks(ax, landmark_list, sign_types, 2*arrow_width)

    position_car, orientation_car = gt_estimator.get_pose(float(1261230000.908327), method='cubic')
    ax.scatter(position_car[0], position_car[1], s=200, marker='x', color='red')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    tick_steps = 10.0
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=tick_steps))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tick_steps))
    plt.show()


GPS_MEASUREMENTS_PATH = '07/gps.csv'
IMU_MEASUREMENTS_PATH = '07/imu.csv'
gps_full_data, imu_full_data = load_gps_and_imu_data(GPS_MEASUREMENTS_PATH, IMU_MEASUREMENTS_PATH)
gt_estimator = GroundTruthEstimator(gps_full_data, imu_full_data, print_kf_progress=True)

landmark_list = util.pickle_load('./map_07.pickle')
possible_camera_poses = util.pickle_load('./map_07_possible_poses.pickle')
scores = util.pickle_load('./output/scores/merged/07_right_map.pickle')['img_CAMERA1_1261230000.908327_right.jpg']
detections = util.pickle_load('./detections_07_right.pickle')['img_CAMERA1_1261230000.908327_right.jpg']

# image = cv2.imread('../final/images/localization/img_CAMERA1_1261230000.908327_right.jpg')
# detection.detect_traffic_signs_in_image(image, {detection.TrafficSignType.CROSSING})

camera = get_camera_malaga_extract_07_right()
#pose = np.array([149.,         -12.,           0.,          -0.53446911,   0.46297167, -0.46297167,   0.53446911])
pose = ImagePose(position=np.array([149., -12., 0.]), orientation=np.array([-0.53446911, 0.46297167, -0.46297167, 0.53446911]))
#pose = ImagePose(position=np.array([149., -17., 0.]), orientation=np.array([-0.53446911, 0.46297167, -0.46297167, 0.53446911]))
predicted_detections = prediction.predicted_detections(pose, landmark_list, camera, debug=True)

debug_image = 255 * np.ones((camera.height, camera.width, 3))
debug_image = detection.generate_debug_image(debug_image, predicted_detections)
cv2.imwrite('best_pose_prediction.jpg', debug_image)
plt.imshow(util.bgr_to_rgb(debug_image)/255)
plt.show()

# score_val = score.get_score(predicted_detections, detections, {detection.TrafficSignType.CROSSING}, debug=True)

# show_heatmap(possible_camera_poses, scores, landmark_list, detection.ALL_SIGN_TYPES)
