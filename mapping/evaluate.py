import numpy as np
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera, malaga_car_pose_to_camera_pose
from detection import TrafficSignType, TrafficSignDetection
import util
import sys
import images
from ground_truth_estimator import GroundTruthEstimator
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

np.set_printoptions(threshold=sys.maxsize)


POSES_PATH = "./possible_poses.pickle"
GPS_PATH = "./transform_routes/transf_routes_overlap/routeFull_in_route7coords.npy"
IMU_PATH = "/home/christian/downloads/datasets/malaga-urban-dataset-plain-text/malaga-urban-dataset_IMU.txt"

detections_path_06 = "./detections_06_right.pickle"
detections_path_07 = "./detections_07_right.pickle"
detections_path_08 = "./detections_08_right.pickle"
detections_path_10 = "./detections_10_right.pickle"

scores_path_06 = "./output/scores/merged/06_right.pickle"
scores_path_07_map = "./output/scores/merged/07_right_map.pickle"
scores_path_07_rem = "./output/scores/merged/07_right_remaining.pickle"
scores_path_08 = "./output/scores/merged/08_right.pickle"
scores_path_10 = "./output/scores/merged/10_right.pickle"

query_set_paths = [
    #(detections_path_06, scores_path_06, "06 overlap"), # Removed because that overlap contains no traffic sign
    (detections_path_07, scores_path_07_map, "07 mapping"),
    (detections_path_07, scores_path_07_rem, "07 remaining"),
    (detections_path_08, scores_path_08, "08 overlap"),
    (detections_path_10, scores_path_10, "10 overlap"),
]

def get_rank(poses, scores, gt_pos, top_n, threshold):
    gt_xy = gt_pos[0:2]

    scores_1d = np.reshape(scores, (scores.size,))
    poses_1d = np.reshape(poses, (poses.shape[0]*poses.shape[1]*poses.shape[2],7))
    s = np.asarray([scores_1d])

    # Combine poses and their scores into a single array
    combined = np.hstack((poses_1d[:,0:2], s.T))

    # Sorted by score, highest first
    sorted_idx = np.argsort(combined[:,-1])[::-1]
    sorted_combined = combined[sorted_idx]

    # Leave only (x,y)
    sorted_poses = sorted_combined[:,0:2]
    top_poses = sorted_poses[:top_n]

    errors = np.linalg.norm(top_poses - gt_xy, axis=1)
    over_threshold = (errors < threshold).astype(int)
    precision = np.clip(np.cumsum(over_threshold), 0, 1)

    return precision


def get_ground_truth(image_name, gt_estimator):
    timestamp = images.get_timestamps_from_images([image_name])[0]
    position = gt_estimator.get_position(timestamp, method='cubic')
    return position


def iterate_queries(gt_estimator, possible_poses, detections, scores, detection_cnt, top_n, threshold):
    rank = np.zeros((top_n,))

    count = 0
    for image_name in list(scores.keys()):
        #print(image_name)
        ground_truth_pos = get_ground_truth(image_name, gt_estimator)
        score_arr = scores[image_name]
        detection_list = detections[image_name]

        if len(detection_list) == detection_cnt:
            rank_result = get_rank(possible_poses, score_arr, ground_truth_pos, top_n, threshold)
            rank += rank_result
            count += 1

    if (count != 0):
        rank /= count
        rank *= 100

    return rank, count


if __name__ == '__main__':
    print("Loading general data...")
    possible_poses = util.pickle_load(POSES_PATH)
    gps_full_data = np.load(GPS_PATH)
    imu_full_data = None#np.genfromtxt(IMU_PATH, skip_header=1)
    gt_estimator = GroundTruthEstimator(gps_full_data, imu_full_data, print_kf_progress=True)

    assert(possible_poses is not None)

    print("Done")


    def plot_query_sets(top_n, threshold, detection_cnt):
        print(f'{top_n}-{threshold}-{detection_cnt}')
        fig = plt.figure()
        ax = fig.add_subplot(111)

        legend = []

        for detections_path, scores_path, name in query_set_paths:
            #print("Loading query set data...")
            #print(detections_path)
            #print(scores_path)
            detections = util.pickle_load(detections_path)
            scores = util.pickle_load(scores_path)

            assert(detections is not None)
            assert(scores is not None)

            data, n = iterate_queries(gt_estimator, possible_poses, detections, scores, detection_cnt, top_n, threshold)

            legend.append(f'{name} (n={n})')
            ax.plot(data)

        #ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
        ax.grid()
        ax.set_ylim((0, 100))
        ax.set_xlabel('rank')
        ax.set_ylabel('localized within threshold (%)')
        ax.legend(legend, loc='lower right')
        plt.title(f'Localization precision (detections={detection_cnt}, threshold={threshold} m)')
        plt.savefig(f'./evaluation/png/evaluation-{top_n}-{threshold}-{detection_cnt}.png', bbox_inches='tight')
        plt.savefig(f'./evaluation/svg/evaluation-{top_n}-{threshold}-{detection_cnt}.svg', bbox_inches='tight')
        #plt.show()

    plot_query_sets(100, 5, 1)
    plot_query_sets(100, 10, 1)
    plot_query_sets(100, 15, 1)
    plot_query_sets(100, 20, 1)
    plot_query_sets(100, 25, 1)
    plot_query_sets(100, 50, 1)

    plot_query_sets(500, 5, 1)
    plot_query_sets(500, 10, 1)
    plot_query_sets(500, 15, 1)
    plot_query_sets(500, 20, 1)
    plot_query_sets(500, 25, 1)
    plot_query_sets(500, 50, 1)

    plot_query_sets(100, 5, 2)
    plot_query_sets(100, 10, 2)
    plot_query_sets(100, 15, 2)
    plot_query_sets(100, 20, 2)
    plot_query_sets(100, 25, 2)
    plot_query_sets(100, 50, 2)

    plot_query_sets(500, 5, 2)
    plot_query_sets(500, 10, 2)
    plot_query_sets(500, 15, 2)
    plot_query_sets(500, 20, 2)
    plot_query_sets(500, 25, 2)
    plot_query_sets(500, 50, 2)

    plot_query_sets(100, 5, 3)
    plot_query_sets(100, 10, 3)
    plot_query_sets(100, 15, 3)
    plot_query_sets(100, 20, 3)
    plot_query_sets(100, 25, 3)
    plot_query_sets(100, 50, 3)

    plot_query_sets(500, 5, 3)
    plot_query_sets(500, 10, 3)
    plot_query_sets(500, 15, 3)
    plot_query_sets(500, 20, 3)
    plot_query_sets(500, 25, 3)
    plot_query_sets(500, 50, 3)








