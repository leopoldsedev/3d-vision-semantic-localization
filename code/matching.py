import images
import util
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.cluster import DBSCAN

import detection


FeatureMatch = namedtuple('FeatureMatch', ['image_idx1', 'detection_idx1', 'image_idx2', 'detection_idx2'])


def filter_detections_by_sign_type(detections, sign_type):
    result = {}

    for image_name in detections:
        detections_in_image = detections[image_name]
        result[image_name] = [d for d in detections_in_image if d.sign_type == sign_type]

    return result


def cluster_through_time(image_names, detections, sign_type=None):
    timestamps = images.get_timestamps_from_images(image_names)
    indices = range(len(image_names))

    detection_matrix = []

    for image_idx, image_name, timestamp in zip(indices, image_names, timestamps):
        if image_name not in detections:
            continue

        detections_per_image = detections[image_name]

        for detection_idx, d in enumerate(detections_per_image):
            detection_matrix.append([image_idx, detection_idx, timestamp, d.x, d.y])

    if len(detection_matrix) == 0:
        return []

    detection_matrix = np.array(detection_matrix)
    # Clustering will be done based on time and pixel coordinates simultaneously and an arbitrary relation between those two units must be chosen.
    # TODO This could be made dependent on the velocity at each point if it was available
    time_to_pixel_scaling = 100 # 100 pixel per second
    detection_matrix[:,2] *= time_to_pixel_scaling

    db_scan = DBSCAN(eps=100, min_samples=10)
    db_scan_result = db_scan.fit(detection_matrix[:,2:])
    labels = db_scan_result.labels_

    unique_labels = set(labels)

    matches = []
    for label in unique_labels:
        # Do not include noise in the result
        #if label == -1:
            #continue

        cluster_mask = (labels == label)
        cluster_detections = detection_matrix[cluster_mask]

        if label != -1:
            assert(cluster_detections.shape[0] >= 2)

            base_detection = cluster_detections[0]
            base_image_idx = int(base_detection[0])
            base_detection_idx = int(base_detection[1])
            for row in cluster_detections[1:,:]:
                image_idx = int(row[0])
                detection_idx = int(row[1])
                match = FeatureMatch(image_idx1=base_image_idx, detection_idx1=base_detection_idx, image_idx2=image_idx, detection_idx2=detection_idx)
                matches.append(match)

        c = util.color_tuple_bgr_to_plt(detection.sign_type_colors[sign_type]) if label != -1 else (0,0,0)
        plt.scatter(cluster_detections[:,3], cluster_detections[:,2] / time_to_pixel_scaling, color=c)

    return matches


def match_detections(image_dir_path, detections):
    """
    Matches the given detections between images

    :param image_paths: The list of image paths (in order of the image sequence).
    :param detections: The detection dictionary from detection.detect_traffic_signs().
    :returns: List of instances of FeatureMatch
    """

    result = []

    image_paths = images.get_image_path_list(image_dir_path)

    image_names = images.get_image_names(image_dir_path)
    timestamps = images.get_timestamps_from_images(image_names)

    detected_sign_types = detection.ALL_SIGN_TYPES

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for sign_type in detected_sign_types:
        filtered_detections = filter_detections_by_sign_type(detections, sign_type)

        matches = cluster_through_time(image_names, filtered_detections, sign_type=sign_type)
        result.extend(matches)

        for image_name, timestamp in zip(image_names, timestamps):
            if image_name not in filtered_detections:
                continue

            d = filtered_detections[image_name]
            x = [detection.x for detection in d]
            y = [detection.y for detection in d]
            color = util.color_tuple_bgr_to_plt(detection.sign_type_colors[sign_type])

    ax.set_xlabel('x (px)')
    ax.set_ylabel('time (s)')
    ax.set_xlim((0, 1024))
    ax.grid()
    timespan = np.max(timestamps) - np.min(timestamps)
    ax.set_ylim((np.min(timestamps)-timespan*0.05, np.max(timestamps)+timespan*0.05))
    #plt.show()

    return result
