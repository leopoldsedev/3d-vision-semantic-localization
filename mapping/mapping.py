from os import path

import detection
from detection import TrafficSignDetection, TrafficSignType
import matching
import triangulation
#import map_encoding
import images
from ground_truth_estimator import GroundTruthEstimator


IMAGE_DIR_PATH = './images-10'
#IMAGE_DIR_PATH = './07/images/rectified'
#IMAGE_DIR_PATH = '/home/christian/downloads/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images/right'
GPS_MEASUREMENTS_PATH = '07/gps.csv'
IMU_MEASUREMENTS_PATH = '07/imu.csv'
COLMAP_WORKING_DIR_PATH = 'colmap'
COLMAP_EXECUTABLE_PATH = '/home/christian/downloads/datasets/colmap/colmap/build/src/exe/colmap'

DETECTION_CACHE_PATH = 'detections.pickle'


def print_heading(heading):
    heading_width = 80
    print('')
    print('=' * heading_width)
    print(heading)
    print('=' * heading_width)
    print('')


if __name__ == '__main__':
    print_heading('Feature detection')

    detections = detection.load_detections(DETECTION_CACHE_PATH)
    if detections is None:
        print(f'No saved detections found at \'{DETECTION_CACHE_PATH}\'. Running detection...')
        detections = detection.detect_traffic_signs(IMAGE_DIR_PATH)
        if (len(detections) > 0):
            detection.save_detections(DETECTION_CACHE_PATH, detections)
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
    gt_estimator = GroundTruthEstimator(GPS_MEASUREMENTS_PATH, IMU_MEASUREMENTS_PATH)

    landmark_list = triangulation.triangulate(COLMAP_EXECUTABLE_PATH, IMAGE_DIR_PATH, detections, matches, gt_estimator, COLMAP_WORKING_DIR_PATH)
