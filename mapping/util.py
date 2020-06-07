import numpy as np
import cv2
import pickle
import os


def color_tuple_bgr_to_plt(color_tuple):
    return (color_tuple[2]/255, color_tuple[1]/255, color_tuple[0]/255)


def bgr_to_rgb(image_bgr):
    b, g, r = cv2.split(image_bgr)
    image_rgb = cv2.merge([r, g, b])
    return image_rgb


def pickle_save(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(file_path):
    if not os.path.isfile(file_path):
        return None

    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
