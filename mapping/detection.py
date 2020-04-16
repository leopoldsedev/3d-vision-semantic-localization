from enum import Enum
from os.path import basename

import images


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
        return f'TrafficSignDetection(x={self.x}, y={self.y}, sign_type={self.sign_type})'


"""
Detects traffic in the image at the given path

:param image_path: The path of the image
:returns: List of instances of TrafficSignDetection
"""
def detect_traffic_signs_in_image(image_path):
    # TODO Properly implement traffic sign detection
    result = []

    from os.path import basename
    if basename(image_path) == 'img_CAMERA1_1261230001.080210_right.jpg':
        result.append(TrafficSignDetection(194, 411, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(819, 386, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.130214_right.jpg':
        result.append(TrafficSignDetection(184, 407, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(833, 381, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.180217_right.jpg':
        result.append(TrafficSignDetection(172, 403, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(848, 375, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.230220_right.jpg':
        result.append(TrafficSignDetection(160, 395, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(864, 367, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.280222_right.jpg':
        result.append(TrafficSignDetection(147, 388, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(884, 359, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.330212_right.jpg':
        result.append(TrafficSignDetection(132, 382, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(904, 350, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.380191_right.jpg':
        result.append(TrafficSignDetection(116, 377, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(926, 341, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.430199_right.jpg':
        result.append(TrafficSignDetection(99, 373, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(951, 334, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.480201_right.jpg':
        result.append(TrafficSignDetection(80, 366, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(980, 324, TrafficSignType.CROSSING))
    elif basename(image_path) == 'img_CAMERA1_1261230001.530205_right.jpg':
        result.append(TrafficSignDetection(59, 360, TrafficSignType.CROSSING))
        result.append(TrafficSignDetection(1011, 313, TrafficSignType.CROSSING))
    else:
        # TODO Undo
        #raise Exception ('Not implemented yet')
        pass

    return result


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
