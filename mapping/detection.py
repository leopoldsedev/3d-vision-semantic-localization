from enum import Enum
from os.path import basename
import images
#for detection
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

class TrafficSignType(Enum):
    CROSSING = 1
    ROUNDABOUT = 2
    # TODO Add other types


class TrafficSignDetection():
    def __init__(self, x, y, sign_type):
        self.x = x
        self.y = y
        self.sign_type = sign_type

    def __repr__(self):
        return f'TrafficSignDetection(x={self.x}, y={self.y}, sign_type={self.sign_type}'

#import template(s)
templatePath = '/home/patricia/3D/multiscale-template-matching/multiscale-template-matching/template/yield/keinvorfahrt.jpg'
template = cv2.imread(templatePath)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

"""
Detects traffic in the image at the given path

:param image_path: The path of the image
:returns: List of instances of TrafficSignDetection
"""
maxVals = []
def detect_traffic_signs_in_image(image_path):
    results = []
    image = cv2.imread(image_path)
    alpha = 3.0 # Contrast control (1.0-3.0)
    beta = 20 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.2, 2.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        width = int(gray.shape[1]*scale)
        height = int(gray.shape[0]*scale)
        dim = (width,height)
        resized = cv2.resize(gray, dim, interpolation = cv2.INTER_CUBIC)
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
                break

        # detect edges and apply template
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

    # unpack the bookkeeping varaible and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found
    maxVals.append(maxVal)
    # set threshold for different templates (5000000 for crosswalk, 3200000 for yield, 5200000 for roundabout)
    if maxVal>3200000:
        results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
    else:
        results.append("no match")
        
#        if basename(image_path) == 'img_CAMERA1_1261230008.980277_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261230078.880928_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261230000.780191_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261230001.480201_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261229997.280171_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261230076.880919_left.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261230036.530548_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261230011.280294_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        elif basename(image_path) == 'img_CAMERA1_1261229995.680150_right.jpg':
#            results.append(TrafficSignDetection(maxLoc[0], maxLoc[1], TrafficSignType.CROSSING))
#        else:
#            # TODO Undo
#            #raise Exception ('Not implemented yet')
#            pass
    print(results)

    return results



"""
Detects traffic in the images at the given paths

:param image_paths: A list of paths to images
:returns: A dictionary where the keys are image paths and the values are list of instances of TrafficSignDetection
"""
def detect_traffic_signs(image_dir_path):
    result = {}

    image_paths = images.get_image_path_list(image_dir_path)
    image_count = len(image_paths)
    print('Processing {} images.'.format(image_count))

    for image_path in image_paths:
        image_name = basename(image_path)
        result[image_name] = detect_traffic_signs_in_image(image_path)

    return result

if __name__ == '__main__':
    path = '/home/patricia/3D/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images'
#    path = '/home/patricia/3D/multiscale-template-matching/multiscale-template-matching/malaga/testall'
    
    # save results in a txt file
    f = open("yield.txt","a+")
    results = detect_traffic_signs(path)
    for i,j in results.items():
        f.write("%s %s \n" % (i,j))
    f.close()
    plt.hist(maxVals, density = True, bins = 1000)
    plt.show()
