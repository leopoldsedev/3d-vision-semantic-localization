import glob
import re
import numpy as np
from os.path import join, basename


def get_image_path_list(image_dir_path):
    return sorted(glob.glob(join(image_dir_path, '*.jpg')))#[50:-50] # TODO Undo


def get_image_names(image_dir_path):
    image_paths = get_image_path_list(image_dir_path)
    image_names = list([basename(path) for path in image_paths])
    return image_names


def get_timestamps_from_images(image_names):
    # The timestamps are part of the file names of the images.
    # Examples:
    # img_CAMERA1_1261230001.080210_right.jpg
    # img_CAMERA1_1261230001.080210_left.jpg

    pattern = re.compile('img_CAMERA1_(\d*.\d*)_(right|left).jpg')

    result = []

    for name in image_names:
        match = pattern.match(name)
        timestamp_str = match.group(1)
        timestamp = float(timestamp_str)
        result.append(timestamp)
        assert(str(timestamp) in name)

    # There should be one timestamp for each image
    assert(len(result) == len(image_names))
    # Timestamps need to be monotonic
    assert(np.all(np.diff(np.array(result)) > 0))

    return result
