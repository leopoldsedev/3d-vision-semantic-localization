import pickle
from collections import namedtuple
from detection import TrafficSignDetection, TrafficSignType
import math
import numpy as np

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
ALL_SIGN_TYPES = [TrafficSignType.CROSSING, TrafficSignType.YIELD, TrafficSignType.ROUNDABOUT]

# NOTE: Guassain score is between 0 and 1

def calculate_guassian(sigma,x,y):
    ret = 1/(2 * math.pi * sigma * sigma)
    distance_sq = (x**2) + (y**2)
    exp = -1*distance_sq/(2*sigma*sigma)
    ret *= math.exp(exp)
    return ret
# for this score we convert the bounding boxes into circles
#  keeping the same center and defining the radius as half
# of the diaganol of the box. We create a uniform distribution in the circles
# we then use a guassian distribution on the remaining pixels for each bounding box
# and then we average the values for the remaining pixels.

# Do to the circle transformation this needs to be modified in cases where the bounding box is in the corner or edges
def create_gaussian_score_arr_single_class(detectionSet):
    ret = np.zeros(shape=(IMAGE_WIDTH,IMAGE_HEIGHT))
    temp = np.zeros(shape=(IMAGE_WIDTH,IMAGE_HEIGHT))
    total = 0
    for detection in detectionSet:
        total += 1
        radius = math.sqrt((detection.height**2) + (detection.width**2))/2
        center_x = detection.x
        center_y = detection.y
        uniform_pixel_value = 1/(math.pi * (radius**2))
        for i in range(0,IMAGE_WIDTH):
            for j in range(0,IMAGE_HEIGHT):
                distance = math.sqrt(((i - center_x)**2) + ((j - center_y)**2))
                if distance <= radius:
                    ret[i][j] = uniform_pixel_value
                else:
                    temp[i][j] += calculate_guassian(radius,i - center_x, j - center_y)

    assert(total > 0)

    for i in range(0,IMAGE_WIDTH):
        for j in range(0,IMAGE_HEIGHT):
            if(ret[i][j] == 0):
                ret[i][j] = temp[i][j]/total
    return ret

def calculate_guassian_score_single_class(detectionSet1,detectionSet2):
    ref = create_gaussian_score_arr_single_class(detectionSet1)
    score = 0
    num_detections = 0
    for detection in detectionSet2:
        radius = math.sqrt((detection.height**2) + (detection.width**2))/2
        center_x = detection.x
        center_y = detection.y
        num_detections += 1
        for i in range(int(center_x - radius), int(center_x + radius)):
            if (i >= IMAGE_WIDTH) or (i < 0):
                continue
            for j in range(int(center_y - radius), int(center_y + radius)):
                if(j >= IMAGE_HEIGHT) or (j < 0):
                    continue
                distance = math.sqrt(((center_x - i) ** 2) + ((center_y - j) ** 2))
                if(distance <= radius):
                    score += ref[i][j]
    if(num_detections == 0):
        return 0     
    return score/num_detections

# dectectionSet1 = P
# dectectionSet2 = Q
def get_score(detectionSet1, detectionSet2, sign_types):
    guassian_score = 0
    for sign_type in sign_types:
        set1 = list(filter(lambda d: d.sign_type == sign_type, detectionSet1))
        set2 = list(filter(lambda d: d.sign_type == sign_type, detectionSet2))
        guassian_score += calculate_guassian_score_single_class(set1,set2)

    return guassian_score


if __name__ == '__main__':
    testImg = "img_CAMERA1_1261229988.730080_right.jpg"
    with open('dectections.pkl', 'rb') as dectections:
    
        # Step 3
        detectionsDict = pickle.load(dectections)
        testImgDections = detectionsDict[testImg]
        for img in detectionsDict:
            detectionList = detectionsDict[img]
            if(len(detectionList) == 0):
                continue
            total_score, guassian_score = get_score(testImgDections,detectionList,ALL_SIGN_TYPES)
            print("{} --- {}\n original score{}\n guassian score{}".format(testImg,img,total_score,guassian_score))


    
