import matplotlib.pyplot as plt
import pickle
from collections import namedtuple
from detection import TrafficSignDetection, TrafficSignType
import detection
import math
import numpy as np
from scipy.stats import multivariate_normal
from numpy import interp


IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
ALL_SIGN_TYPES = [TrafficSignType.CROSSING, TrafficSignType.YIELD, TrafficSignType.ROUNDABOUT]


def gaussian_grid(width, height, mean, cov):
    """
    Creates a guassian matrix by applying a multivariate guassian

    :param height: height of image
    :param width: width of image
    :param mean: center of bounding box and mean for guassian
    :param cov: covariance matrix
    :returns: numpy array representing guassian matrix
    """
    var = multivariate_normal(mean=mean, cov=cov)

    x, y = np.meshgrid(range(width), range(height))
    grid = np.dstack((x,y))

    # var.pdf() takes a list of points, which is why we need to reshape the
    # input and then reshape the output back into the grid shape
    sample_coordinates = np.reshape(grid, (width*height, 2))
    assert(np.allclose(np.dstack((x,y)), np.reshape(sample_coordinates, (height, width, 2))))

    samples = np.reshape(var.pdf(sample_coordinates), (height, width)).T
    assert(samples.shape[0] == width)
    assert(samples.shape[1] == height)

    return samples


def create_gaussian_score_arr(detectionSet):
    """
    Create a numpy array which corresonds to a guassian score for a given detection set.
    We downsample to increase performance.

    :param detectionSet: a list of detections
    :returns: numpy array with dimensions corresponding to the
            IMAGE_WIDTH and IMAGE_HEIGHT represents the
            guassian score for the detection set.
    """
    array_width = 20
    scale = array_width / IMAGE_WIDTH
    array_height = int(scale * IMAGE_HEIGHT)
    ret = np.zeros(shape=(array_width, array_height))

    for detection in detectionSet:
        mean = np.array([detection.x, detection.y]) * scale

        deviation_x = detection.width
        deviation_y = detection.height
        deviation = np.array([deviation_x, deviation_y]) * scale
        cov = np.diag(10*(deviation**2))

        distribution = gaussian_grid(array_width, array_height, mean, cov)
        ret += distribution

    return ret


def calculate_gaussian_match_score(detections_predicted, detections_query, debug=False):
    """
    Calculates the guassian match score

    :param detections_predicted: list dettection objects derived from the enviroment
    :param detections_query: list of dettection objects derived from the query image
    :param debug: boolean indicating whether or not debug output is desired
    :returns: float score representing the guassian match score
    """
    gaussian_query = create_gaussian_score_arr(detections_query)
    gaussian_predicted = create_gaussian_score_arr(detections_predicted)

    correlation = gaussian_query * gaussian_predicted

    # Source for upper bound: https://math.stackexchange.com/a/22579
    max_total_correlation = np.sqrt(np.sum(gaussian_query**2) * np.sum(gaussian_predicted**2))
    total_correlation = np.sum(correlation)
    if max_total_correlation == 0.0:
        total_correlation_normalized = 0.0
    else:
        total_correlation_normalized = np.sum(correlation) / max_total_correlation

    diff = np.abs(gaussian_query - gaussian_predicted)

    max_total_diff = np.sum(gaussian_query) + np.sum(gaussian_predicted)
    total_diff = np.sum(diff)
    if max_total_diff == 0.0:
        total_diff_normalized = 0.0
    else:
        total_diff_normalized = total_diff / max_total_diff

    assert(0 <= total_correlation_normalized <= 1)
    assert(0 <= total_diff_normalized <= 1)

    match_score = total_correlation_normalized - total_diff_normalized
    match_score = interp(match_score, [-1,1], [0,1])

    if debug:
        print('-----------')
        print(f'total_correlation={total_correlation}')
        print(f'max_total_correlation={max_total_correlation}')
        print(f'total_diff={total_diff}')
        print(f'max_total_diff={max_total_diff}')
        print(f'match_score={match_score}')

        show_distribution('query', gaussian_query, None, None)
        show_distribution('predicted', gaussian_predicted, None, None)
        show_distribution('correlation', correlation / max_total_correlation, 0.0, None)
        show_distribution('difference', diff / max_total_diff, 0.0, None)

    return match_score


def show_distribution(title, distribution, low, high):
    """
    Visualizes the guassian distribution

    :param high: upper bound value for the graph, can be set to None
    :param low: lower bound value for the graph, can be set to None
    """
    #print(f'sum={np.sum(distribution)}')
    im = plt.imshow(distribution.T)
    if low is not None and high is not None:
        im.set_clim(low, high)
    plt.colorbar()
    plt.title(title)
    plt.show()


def get_score(detections_predicted, detections_query, sign_types, debug=False):
    """
    Calculates final match score

    :param detections_predicted: list dettection objects derived from the enviroment
    :param detections_query: list of dettection objects derived from the query image
    :sign_types: list of containing the sign types
    :returns: float score representing the final match score
    """
    gaussian_score = 0
    for sign_type in sign_types:
        if debug:
            print(f'Calculating score for sign type \'{sign_type}\'')
        set1 = list(filter(lambda d: d.sign_type == sign_type, detections_predicted))
        set2 = list(filter(lambda d: d.sign_type == sign_type, detections_query))

        gaussian_score += calculate_gaussian_match_score(set1, set2, debug)

    average_gaussian_score = gaussian_score / len(sign_types)

    return average_gaussian_score


if __name__ == '__main__':
    ALL_SIGN_TYPES = [TrafficSignType.CROSSING, TrafficSignType.YIELD, TrafficSignType.ROUNDABOUT]
    ALL_SIGN_TYPES = [TrafficSignType.CROSSING]

    detectionPredicted1 = detection.TrafficSignDetection(x=438, y=223, width=40, height=40, sign_type=detection.TrafficSignType.CROSSING, score=0)
    detectionPredicted2 = detection.TrafficSignDetection(x=800, y=500, width=40, height=40, sign_type=detection.TrafficSignType.CROSSING, score=0)
    detectionPredicted3 = detection.TrafficSignDetection(x=0, y=0, width=40, height=40, sign_type=detection.TrafficSignType.CROSSING, score=0)
    detectionPredicted4 = detection.TrafficSignDetection(x=0, y=0, width=80, height=80, sign_type=detection.TrafficSignType.CROSSING, score=0)
    detectionPredicted5 = detection.TrafficSignDetection(x=438, y=223, width=80, height=80, sign_type=detection.TrafficSignType.CROSSING, score=0)

    detectionQuery1 = detection.TrafficSignDetection(x=438, y=223, width=40, height=40, sign_type=detection.TrafficSignType.CROSSING, score=0)
    detectionQuery2 = detection.TrafficSignDetection(x=500, y=300, width=40, height=40, sign_type=detection.TrafficSignType.CROSSING, score=0)

    # Should go from highest to lowest score
    predicted = [detectionPredicted1, detectionPredicted2]
    query = [detectionQuery1, detectionPredicted2]
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # >
    predicted = [detectionPredicted1]
    query = [detectionQuery1]
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # >
    predicted = []
    query = []
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # > (?)
    predicted = [detectionPredicted5]
    query = [detectionQuery1]
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # >
    predicted = [detectionPredicted1, detectionPredicted2, detectionPredicted3]
    query = [detectionQuery1]
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # > (?)
    predicted = [detectionPredicted1]
    query = [detectionQuery1, detectionQuery2]
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # > (?)
    predicted = [detectionPredicted2]
    query = [detectionQuery1]
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # >
    predicted = []
    query = [detectionQuery1]
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # ==
    print('equal')
    predicted = [detectionPredicted1]
    query = []
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
    # == (?)
    predicted = [detectionPredicted5]
    query = []
    print(predicted)
    print(query)
    gaussian_score = get_score(predicted, query, ALL_SIGN_TYPES, debug=True)
    print(f'gaussian_score={gaussian_score}')
