import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import cv2
import transforms3d as tf
from numpy import interp
from os.path import basename, join

import prediction
import score
import detection
import mapping
import util
import images
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera, malaga_car_pose_to_camera_pose
from detection import TrafficSignDetection, TrafficSignType
from ground_truth_estimator import GroundTruthEstimator

MAP_PATH = './output/map.pickle'
GPS_PATH = './data/transformed_routes_gps/full_frame07.npy'

QUERY_IMAGE_PATH = './data/query images/img_CAMERA1_1261230073.830899_right.jpg'
#QUERY_IMAGE_PATH = './data/query images/img_CAMERA1_1261230074.880882_right.jpg'
#QUERY_IMAGE_PATH = './data/query images/img_CAMERA1_1261230076.380893_right.jpg'
#QUERY_IMAGE_PATH = './data/query images/img_CAMERA1_1261230079.630932_right.jpg'

POSITION_STEP_SIZE = 5
ANGLE_STEP_SIZE = 10
LANDMARK_MARGIN = prediction.LANDMARK_RELEVANCE_RANGE


def get_possible_poses(landmark_list, position_step_size, angle_step_size, landmark_margin):
    """
    Depending on how accurate we want the poses to be presented, calculate several possible camera poses

    :param landmark_list: List of landmarks detected in the map
    :param position_step_size: Variable to control definition of the position prediction
    :param angle_step_size: Variable to control definition of the orientation prediction
    :param landmark_margin: Distance at which a landmark is assumed to be visible to the camera
    :returns: Array of possible poses and orientation
    """
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
    """
    Attach scores to each possible pose from "get_possible_poses"

    :param landmark_list: List of landmarks detected in the map
    :param query_detections: List of instances of TrafficSignDetection
    :param possible_camera_poses: Possible poses and orientation of the camera
    :param camera: Camera parameters
    :param sign_types: List of landmark types
    :returns: Array where scores are attached to possible poses 
    """

    # Filter only landmarks of given sign types
    landmark_list = list(filter(lambda l: l.sign_type in sign_types, landmark_list))

    empty_predicted_score = score.get_score([], query_detections, sign_types, debug=False)

    total_poses = possible_camera_poses.size / 7
    global score_calc_count
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
            print(np.rad2deg(yaw))
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

    return scores


def split_pose_array(pose_arr):
    assert(pose_arr.shape == (7,))
    position = pose_arr[0:3]
    orientation = pose_arr[3:7]
    return position, orientation


def visualize_landmarks(ax, landmark_list, sign_types, landmark_arrow_width):
    """
    :param ax: Subplot of the figure
    :param landmark_list: List of landmarks detected in the map
    :param sign_types: List of landmark types
    :param landmark_arrow_width: Width of landmark arrow that points towards the orientation
    """
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


def show_heatmap(possible_poses, pose_scores, landmark_list, sign_types, actual_position):
    """
    :param possible_poses: Possible poses and orientation of the camera
    :param pose_scores: Scores calculated from `get_pose_scores'
    :param landmark_list: List of landmarks detected in the map
    :param sign_types: List of landmark types
    :param actual_position: Actual camera position from the ground truth
    :returns: Visualization of heatmap with comparison to actual location marked as a red cross
    """
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

    visualize_landmarks(ax, landmark_list, sign_types, 2*arrow_width)

    # Visualize actual position
    ax.scatter(actual_position[0], actual_position[1], s=200, marker='x', color='red')

    tick_steps = 10.0
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=tick_steps))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tick_steps))

    plt.show()


if __name__ == '__main__':
    landmark_list = util.pickle_load(MAP_PATH)
    query_image = cv2.imread(QUERY_IMAGE_PATH)
    assert(query_image is not None)
    query_image_name = basename(QUERY_IMAGE_PATH)

    camera = get_camera_malaga_extract_07_right()

    sign_types = detection.ALL_SIGN_TYPES
    print('Detecting traffic signs in query image...')
    query_detections, detection_debug_image = detection.detect_traffic_signs_in_image(query_image, sign_types)
    plt.figure()
    plt.imshow(util.bgr_to_rgb(detection_debug_image))
    plt.show()

    print('Calculating possible poses...')
    possible_camera_poses = get_possible_poses(landmark_list, POSITION_STEP_SIZE, ANGLE_STEP_SIZE, LANDMARK_MARGIN)

    print('Calculating scores...')
    pose_scores = get_pose_scores(landmark_list, query_detections, possible_camera_poses, camera, sign_types)

    print('Getting ground truth for query image')
    gps_full_data = np.load(GPS_PATH)
    imu_full_data = None
    gt_estimator = GroundTruthEstimator(gps_full_data, imu_full_data, print_kf_progress=True)
    query_timestamp = images.get_timestamps_from_images([query_image_name])[0]
    actual_position = gt_estimator.get_position(query_timestamp, method='cubic')

    print('Showing heatmap...')
    show_heatmap(possible_camera_poses, pose_scores, landmark_list, sign_types, actual_position)
