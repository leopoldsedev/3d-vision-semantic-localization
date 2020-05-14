import numpy as np
import transforms3d as tf
import matplotlib.pyplot as plt

import util
import detection
from detection import TrafficSignType, TrafficSignDetection
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera
# The distance at which a landmark is assumed to be visible for the camera
LANDMARK_RELEVANCE_RANGE = 30.0

# The angle at which a landmark is assumed to be visible for the camera
# 0 degree means the camera has to face the landmark's direction vector
# exactly, 180 degree means the direction of the landmark does not matter
LANDMARK_RELEVANCE_ANGLE = 90.0


def is_detection_in_image(detection, width, height):
    def is_point_in_image(x, y):
        return (0 <= x <= width) and (0 <= y <= height)

    x1 = detection.x - detection.width / 2.0
    x2 = detection.x + detection.width / 2.0
    y1 = detection.y - detection.height / 2.0
    y2 = detection.y + detection.height / 2.0

    return (
           is_point_in_image(x1, y1)
        or is_point_in_image(x1, y2)
        or is_point_in_image(x2, y1)
        or is_point_in_image(x2, y2)
    )


def is_facing_camera(landmark_cam_frame, camera_pose):
    position = np.array([landmark_cam_frame.x, landmark_cam_frame.y, landmark_cam_frame.z])
    direction = landmark_cam_frame.direction

    position_normalized = position / np.linalg.norm(position)
    # direction vector is assumed to be already normalized

    # Vectors in this formula need to be normalized
    angle = np.arccos(np.dot(position_normalized, direction))

    is_facing_camera = np.rad2deg(angle) > (180 - LANDMARK_RELEVANCE_ANGLE)
    return is_facing_camera


def predict_detection(landmark_cam_frame, camera):
    xyz = np.array([landmark_cam_frame.x, landmark_cam_frame.y, landmark_cam_frame.z])
    center_pixel = project3dToPixel(camera, xyz)

    width_real_world, height_real_world = detection.sign_type_dimensions[landmark_cam_frame.sign_type]

    width_xyz1 = xyz - np.array([-width_real_world/2, 0.0, 0.0])
    width_xyz2 = xyz + np.array([-width_real_world/2, 0.0, 0.0])

    height_xyz1 = xyz - np.array([0.0, -height_real_world/2, 0.0])
    height_xyz2 = xyz + np.array([0.0, -height_real_world/2, 0.0])

    assert(np.allclose(np.linalg.norm(width_xyz2 - width_xyz1), width_real_world))
    assert(np.allclose(np.linalg.norm(height_xyz2 - height_xyz1), height_real_world))

    width_pixel1 = project3dToPixel(camera, width_xyz1)
    width_pixel2 = project3dToPixel(camera, width_xyz2)
    height_pixel1 = project3dToPixel(camera, height_xyz1)
    height_pixel2 = project3dToPixel(camera, height_xyz2)

    predicted_width = np.linalg.norm(width_pixel1 - width_pixel2)
    predicted_height = np.linalg.norm(height_pixel1 - height_pixel2)

    predicted_detection = TrafficSignDetection(x=center_pixel[0], y=center_pixel[1], width=predicted_width, height=predicted_height, sign_type=landmark_cam_frame.sign_type, score=0)

    return predicted_detection


def project3dToPixel(camera, xyz):
    Tx = 0
    Ty = 0
    fx = camera.params[0]
    fy = camera.params[1]
    cx = camera.params[2]
    cy = camera.params[3]
    u = (fx * xyz[0] + Tx) / xyz[2] + cx
    v = (fy * xyz[1] + Ty) / xyz[2] + cy
    return np.array([u, v])


def landmark_map_to_cam_frame(landmark, camera_pose):
    pos = np.array([landmark.x, landmark.y, landmark.z])
    direction = landmark.direction

    # There are 2 relevant coordinate frames:
    # M ... map frame
    # C ... camera frame
    # landmark contains the point in frame M
    # camera_pose contains the translational and rotational part of T_MC
    R_MC = tf.quaternions.quat2mat(camera_pose.orientation)
    T_MC = tf.affines.compose(camera_pose.position, R_MC, np.ones(3))

    # The position vector needs a full homogeneous transformation, the
    # direction vector needs ONLY rotation transformation

    # Convert position vector into homogeneous coordinates
    pos_map_frame = np.array([pos[0], pos[1], pos[2], 1.0])
    # Compute T_CM^(-1) * pos_map_frame
    pos_cam_frame = np.linalg.solve(T_MC, pos_map_frame)

    dir_map_frame = np.array([direction[0], direction[1], direction[2]])
    # Compute R_CM^(-1) * dir_map_frame
    dir_cam_frame = np.linalg.solve(R_MC, dir_map_frame)

    assert(pos_cam_frame[3] == 1.0)
    x_cam_frame = pos_cam_frame[0]
    y_cam_frame = pos_cam_frame[1]
    z_cam_frame = pos_cam_frame[2]

    return landmark._replace(x=x_cam_frame, y=y_cam_frame, z=z_cam_frame, direction=dir_cam_frame)


def predicted_detections(camera_pose, landmark_list, camera, debug=False):
    result = []

    for landmark in landmark_list:
        pos_2d_landmark = np.array([landmark.x, landmark.y])
        pos_2d_camera = camera_pose.position[0:2]

        distance = np.linalg.norm(pos_2d_landmark - pos_2d_camera)

        if debug:
            print(f'pose: {camera_pose}')
            print(f'Checking landmark {landmark}')

        # Check if landmark is in assumed visibility range
        if distance > LANDMARK_RELEVANCE_RANGE:
            if debug:
                print(f'-> not in range')
            continue

        landmark_cam_frame = landmark_map_to_cam_frame(landmark, camera_pose)

        # Check if landmark is in front of camera
        if landmark_cam_frame.z <= 0:
            if debug:
                print(f'-> not in front of camera')
            continue

        # Check if landmark is facing the camera
        if landmark.direction.any() and not is_facing_camera(landmark_cam_frame, camera_pose):
            if debug:
                print(f'-> not facing camera')
            continue

        predicted_detection = predict_detection(landmark_cam_frame, camera)

        # Check if landmark is in image boundaries
        if is_detection_in_image(predicted_detection, camera.width, camera.height):
            result.append(predicted_detection)
            if debug:
                print('-> OKAY')
        else:
            if debug:
                print(predicted_detection)
                print(f'-> not in image boundaries')

    return result


if __name__ == '__main__':
    landmark1 = MapLandmark(x=0, y=3, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark2 = MapLandmark(x=0, y=3, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark3 = MapLandmark(x=1, y=3, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark4 = MapLandmark(x=1, y=3, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark5 = MapLandmark(x=0, y=4, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark6 = MapLandmark(x=0, y=4, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark7 = MapLandmark(x=1, y=4, z=0, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark8 = MapLandmark(x=1, y=4, z=1, sign_type=TrafficSignType.CROSSING, confidence_score=0, direction=np.array([0.0, -1.0, 0.0]))
    landmark_list = [landmark1, landmark2, landmark3, landmark4, landmark5, landmark6, landmark7, landmark8]

    #landmark1 = MapLandmark(x=115.0245186706155, y=-13.501321166892042, z=1.087175120588432, sign_type=TrafficSignType.CROSSING, confidence_score=0.07502497526300567, direction=np.array([0., 0., 0.]))

    #landmark_list = [landmark1]

    #landmark_list = util.pickle_load('./map.pickle')
    #landmark_list = list(filter(lambda l: l.sign_type == TrafficSignType.CROSSING, landmark_list))

    print(landmark_list)

    cam_pos = np.array([150, -11, 0.0])
    cam_pos = np.array([0, 0, 0])
    yaw_deg = -90
    yaw_deg = 0
    cam_rot = tf.euler.euler2mat(np.deg2rad(-90), 0, np.deg2rad(yaw_deg), 'sxyz')
    pose = ImagePose(orientation=tf.quaternions.mat2quat(cam_rot), position=cam_pos)
    #pose = ImagePose(position=np.array([170.,  19.,   0.]), orientation=np.array([ 4.62826766e-17, -4.00913121e-17, -6.54740814e-01,  7.55853469e-01]))
    print(pose)

    camera = get_camera_malaga_extract_07_right()

    detections_predicted = predicted_detections(pose, landmark_list, camera, debug=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for detection in detections_predicted:
        print(detection)
        ax.scatter(detection.x, detection.y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.set_xlim((0, 1024))
    ax.set_ylim((768, 0))
    ax.grid(True)
    plt.show()
