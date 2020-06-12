import numpy as np
from triangulation import MapLandmark, ImagePose, get_camera_malaga_extract_07_right, ColmapCamera, malaga_car_pose_to_camera_pose
from detection import TrafficSignType, TrafficSignDetection
import util
import sys
import images
import localization
from ground_truth_estimator import GroundTruthEstimator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

matplotlib.rcParams['text.usetex'] = True
np.set_printoptions(threshold=sys.maxsize)


POSES_PATH = "./data/map_07_possible_poses.pickle"
GPS_PATH = "./data/transformed_routes_gps/full_frame07.npy"
IMU_PATH = "/home/christian/downloads/datasets/malaga-urban-dataset-plain-text/malaga-urban-dataset_IMU.txt"

detections_path_06 = "./data/detections_06_right.pickle"
detections_path_07 = "./data/detections_07_right.pickle"
detections_path_08 = "./data/detections_08_right.pickle"
detections_path_10 = "./data/detections_10_right.pickle"

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

def get_rank(poses, scores, gt_pos, top_n):
    """
    This function sorts poses from high to low scores
    Then calculates the error in comparison to the ground truth

    :param poses: Array of possible camera poses
    :param scores: Array of scores associated with each pose
    :param top_n: Number of scores from high to low to be taken in part of the evaluation
    :returns: Array of cumulative minimum error between the predicted poses and ground truth
    """

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
    return np.minimum.accumulate(errors)


def get_ground_truth(image_name, gt_estimator):
    timestamp = images.get_timestamps_from_images([image_name])[0]
    position = gt_estimator.get_position(timestamp, method='cubic')
    return position


def iterate_queries(gt_estimator, possible_poses, detections, scores, detection_cnt, top_n):
    """
    Iterate through a set of query images

    :param gt_estimator: Ground truth from GroundTruthEstimator
    :param possible_poses: Array of possible camera poses
    :param detections: List of detections in an image
    :param scores: Scores associated with each pose
    :param detection_cnt: Number of traffic signs detected in an image
    :param top_n: Number of scores from high to low to be taken in part of the evaluation
    :returns rank: Array of cumulative minimum error between the predicted poses and ground truth
    :returns count: Number of query images that match the number of detections in each image
    """
    count = 0
    for image_name in list(scores.keys()):
        detection_list = detections[image_name]

        if len(detection_list) == detection_cnt:
            count += 1

    rank = np.zeros((count,top_n))

    i = 0
    for image_name in list(scores.keys()):
        #print(image_name)
        ground_truth_pos = get_ground_truth(image_name, gt_estimator)
        score_arr = scores[image_name]
        detection_list = detections[image_name]

        if len(detection_list) == detection_cnt:
            rank_result = get_rank(possible_poses, score_arr, ground_truth_pos, top_n)
            rank[i,:] = rank_result
            i += 1

    return rank, count


if __name__ == '__main__':
    print("Loading general data...")
    possible_poses = util.pickle_load(POSES_PATH)
    gps_full_data = np.load(GPS_PATH)
    imu_full_data = None#np.genfromtxt(IMU_PATH, skip_header=1)
    gt_estimator = GroundTruthEstimator(gps_full_data, imu_full_data, print_kf_progress=True)

    assert(possible_poses is not None)

    print("Done")


    def plot_query_sets(top_n, detection_cnt, quantile_size):
        """
        Plots and saves evaluation as .svg and .png files in "./output/evaluation/" folder

        :param top_n: Number of scores from high to low to be taken in part of the evaluation
        :param detection_cnt: Number of traffic signs detected in an image
        :param quantile_size: Size of quantile
        """

        print(f'{top_n}-{detection_cnt}')
        fig = plt.figure()
        ax = fig.add_subplot(111)

        quantile_q = int(1/quantile_size)

        ax.axhline(y=localization.POSITION_STEP_SIZE, linestyle='--', color='black', label='$\\Delta P$')

        for detections_path, scores_path, name in query_set_paths:
            detections = util.pickle_load(detections_path)
            scores = util.pickle_load(scores_path)

            assert(detections is not None)
            assert(scores is not None)

            data, n = iterate_queries(gt_estimator, possible_poses, detections, scores, detection_cnt, top_n)

            print(data.shape)
            median = np.median(data, axis=0)
            lower_quantile = np.quantile(data, quantile_size, axis=0)
            upper_quantile = np.quantile(data, 1 - quantile_size, axis=0)

            color = next(ax._get_lines.prop_cycler)['color']
            x = np.array(range(data.shape[1]))
            ax.plot(x, median, color=color, label=f'{name} ($n$={n})')
            ax.plot(x, lower_quantile, '--', linewidth=1, alpha=0.5, color=color, label=None)
            ax.plot(x, upper_quantile, '--', linewidth=1, alpha=0.5, color=color, label=None)
            #ax.fill_between(x, lower_quantile, upper_quantile, alpha=0.2)

        #ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
        ax.grid()
        ax.set_ylim((0, 50))
        ax.set_xlabel('rank')
        ax.set_ylabel('median min. localization error (m)')
        ax.legend(loc='upper right')
        plural_s = 's' if detection_cnt > 1 else ''
        plt.suptitle(f'Localization accuracy (queries with {detection_cnt} detection{plural_s})', y=0.96)
        plt.title('$\\Delta P = 5$ m, $\\Delta\\theta = 10^{{\\circ}}$, {quantile_q}-quantiles'.format(detection_cnt=detection_cnt, quantile_q=quantile_q), fontsize=10)
        plt.savefig(f'./output/evaluation/png/evaluation-{top_n}-{detection_cnt}.png', bbox_inches='tight')
        plt.savefig(f'./output/evaluation/svg/evaluation-{top_n}-{detection_cnt}.svg', bbox_inches='tight')
        plt.show()


    plot_query_sets(50, 1, 0.25)
    plot_query_sets(50, 2, 0.25)
    plot_query_sets(50, 3, 0.25)
