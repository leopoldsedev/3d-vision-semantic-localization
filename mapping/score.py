# Assumes that for a single class that no rectangles overlap in an image. 
# Use non-maximal suppression if this beccomes a problem.
# normalized for overlapping area
def check_no_overlap_width(detection1,detection2):
    if(detection1.x > detection2.x):
        return get_overlap_area_from_rectangles(detection2, detection1)
    if(detection1.x + detection1.width <= detection2.x):
        return False
    else:
        return True

def check_no_overlap_height(detection1,detection2):
    if(detection1.y > detection2.y):
        return get_overlap_area_from_rectangles(detection2, detection1)
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
    l1x = detection1.x + detection1.width
    l2x = detection2.x + detection2.width   
    l1y = detection1.y + detection1.height
    l2y = detection2.y + detection2.height
    width = min(r1x,r2x) - max(l1x,l2x)
    height = min(r1y,r2y) - max(l1y,l2y)
    return width * height

def calc_score_of_same_class(detectionSet1,detectionSet2):
    totalArea = 0
    overlap = 0
    for detection in detectionSet1:
        totalArea = totalArea + detection1.width*detection1.height
    for detection in detectionSet2:
        totalArea = totalArea + detection2.width*detection2.height
    if totalArea == 0:
        return 0
    for detection1 in detectionSet1:
        for detection2 in detectionSet2:
           overlap = overlap + get_overlap_area_from_rectangles(detection1, detection2)
    return (overlap/totalArea) - ((totalArea - (overlap*2))/totalArea)

# closure factory for filter function, works in OCaml hopefully it does here
def make_in_class(i):
    def in_class(detection): 
        if (detection.sign_type == i): 
            return True
        else: 
            return False
    return in_class
def sum_scores(detectionSet1,detectionSet2,numClasses):
    total_score = 0
    for i in range(1,numClasses+1):
        set1 = filter(make_in_class(i),detectionSet1)
        set2 = filter(make_in_class(i),detectionSet2)
        total_score = total_score + calc_score_of_same_class(set1,set2)
    return total_score




    
