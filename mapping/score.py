import pickle
from collections import namedtuple
from detection import TrafficSignDetection, TrafficSignType
import math
import numpy as np
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768

# NOTE: Guassain score is between 0 and 1

# Assumes that for a single class that no rectangles overlap in an image. 
# Use non-maximal suppression if this beccomes a problem.
# normalized for overlapping area
def check_no_overlap_width(detection1,detection2):
    if(detection1.x > detection2.x):
        return check_no_overlap_width(detection2, detection1)
    if(detection1.x + detection1.width <= detection2.x):
        return False
    else:
        return True

def check_no_overlap_height(detection1,detection2):
    if(detection1.y > detection2.y):
        return check_no_overlap_height(detection2, detection1)
    if(detection1.y + detection1.height <= detection2.y):
        return False
    else:
        return True

def get_overlap_area_from_rectangles(detection1, detection2):
    if not check_no_overlap_height(detection1,detection2):
        return 0
    if not check_no_overlap_width(detection1,detection2):
        return 0
    r1x = detection1.x + detection1.width
    r2x = detection2.x + detection2.width
    r1y = detection1.y + detection1.height
    r2y = detection2.y + detection2.height
    l1x = detection1.x
    l2x = detection2.x
    l1y = detection1.y
    l2y = detection2.y
    width = min(r1x,r2x) - max(l1x,l2x)
    height = min(r1y,r2y) - max(l1y,l2y)
    return width * height

def calc_score_of_same_class(detectionSet1,detectionSet2):
    totalArea = 0
    overlap = 0
    for detection in detectionSet1:
        totalArea = totalArea + detection.width*detection.height
    for detection in detectionSet2:
        totalArea = totalArea + detection.width*detection.height
    if totalArea == 0:
        return 0
    for detection1 in detectionSet1:
        for detection2 in detectionSet2:
           overlap = overlap + get_overlap_area_from_rectangles(detection1, detection2)
    totalArea = totalArea - overlap
    A = overlap
    B = totalArea - overlap
    return (A - B)

# closure factory for filter function, works in OCaml hopefully it does here
def filter(set,className):
    ret = []
    for s in set:
        if s.sign_type == className:
            ret.append(s)
    return ret

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
        center_x = detection.x + (detection.width/2)
        center_y = detection.y + (detection.height/2)
        uniform_pixel_value = 1/(math.pi * (radius**2))
        for i in range(0,IMAGE_WIDTH):
            for j in range(0,IMAGE_HEIGHT):
                distance = math.sqrt(((i - center_x)**2) + ((j - center_y)**2))
                if distance <= radius:
                    ret[i][j] = uniform_pixel_value
                else:
                    temp[i][j] += calculate_guassian(radius,i - center_x, j - center_y)
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
        center_x = detection.x + (detection.width/2)
        center_y = detection.y + (detection.height/2)
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
def get_score(detectionSet1,detectionSet2,classes):
    total_score = 0
    guassian_score = 0
    for i in classes:
        set1 = filter(detectionSet1,i)
        set2 = filter(detectionSet2,i)
        total_score = total_score + calc_score_of_same_class(set1,set2)
        guassian_score += calculate_guassian_score_single_class(set1,set2)
    return total_score, guassian_score


if __name__ == '__main__':
    testImg = "img_CAMERA1_1261229988.730080_right.jpg"
    ALL_SIGN_TYPES = [TrafficSignType.CROSSING, TrafficSignType.YIELD, TrafficSignType.ROUNDABOUT]
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


    
