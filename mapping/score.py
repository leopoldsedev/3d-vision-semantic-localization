import pickle
from collections import namedtuple
from detection import TrafficSignDetection, TrafficSignType

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

def get_score(detectionSet1,detectionSet2,classes):
    total_score = 0
    for i in classes:
        set1 = filter(detectionSet1,i)
        set2 = filter(detectionSet2,i)
        total_score = total_score + calc_score_of_same_class(set1,set2)
    return total_score


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
            score = get_score(testImgDections,detectionList,ALL_SIGN_TYPES)
            print("{} --- {}\n{}".format(testImg,img,score))


    
