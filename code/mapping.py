"""
Module that should be run to create the landmark map.
"""
import numpy as np
import detection
import matching
import triangulation
import util
import images
from detection import TrafficSignDetection, TrafficSignType
from ground_truth_estimator import GroundTruthEstimator, load_gps_and_imu_data


IMAGE_DIR_PATH = './data/mapping images'
GPS_MEASUREMENTS_PATH = './data/transformed_routes_gps/full_frame07.npy'
COLMAP_WORKING_DIR_PATH = './output/colmap'
# TODO Set this path
COLMAP_EXECUTABLE_PATH = '/path/to/colmap'

DETECTION_CACHE_PATH = './output/detections.pickle'
DETECTION_DEBUG_PATH = './output/detection_debug'
MAP_OUTPUT_PATH = './output/map.pickle'


def print_heading(heading):
    heading_width = 80
    print('')
    print('=' * heading_width)
    print(heading)
    print('=' * heading_width)
    print('')


if __name__ == '__main__':
    print_heading('Feature detection')

    # Check if there are cached detections available.
    detections = util.pickle_load(DETECTION_CACHE_PATH)
    if detections is None:
        print(f'No saved detections found at \'{DETECTION_CACHE_PATH}\'. Running detection...')
        detections = detection.detect_traffic_signs(IMAGE_DIR_PATH, debug_output_path=DETECTION_DEBUG_PATH)
        if (len(detections) > 0):
            util.pickle_save(DETECTION_CACHE_PATH, detections)
    else:
        print(f'Loaded detections from \'{DETECTION_CACHE_PATH}\'.\n')

    detection_count = sum([len(detections[x]) for x in detections])
    print(f'Detected {detection_count} traffic signs in total.')


    print_heading('Feature matching')

    matches = matching.match_detections(IMAGE_DIR_PATH, detections)

    match_count = len(matches)
    print(f'Found {match_count} matches.')
    # TODO Print how many match clusters there are


    print_heading('Point triangulation')

    gps_full_data = np.load(GPS_MEASUREMENTS_PATH)
    imu_full_data = None
    gt_estimator = GroundTruthEstimator(gps_full_data, imu_full_data, print_kf_progress=True)

    landmark_list = triangulation.triangulate(COLMAP_EXECUTABLE_PATH, IMAGE_DIR_PATH, detections, matches, gt_estimator, COLMAP_WORKING_DIR_PATH)

    # Save landmark map
    util.pickle_save(MAP_OUTPUT_PATH, landmark_list)
