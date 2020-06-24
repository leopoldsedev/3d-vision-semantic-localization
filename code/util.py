"""
Module containing general utility functions.
"""
import numpy as np
import cv2
import pickle
import os


def color_tuple_bgr_to_plt(color_tuple):
    """
    Convert a BGR color tuple (e.g. (0, 147, 255)) to a color tuple that can be used by Matplotlib (e.g. (0, 0.5764705882352941, 1)).

    :param color_tuple: BGR color tuple.
    :returns: Matplotlib color tuple.
    """
    return (color_tuple[2]/255, color_tuple[1]/255, color_tuple[0]/255)


def bgr_to_rgb(image_bgr):
    """
    Convert a BGR color tuple (e.g. (0, 147, 255)) to an RGB color tuple (e.g. (255, 147, 0)).

    :param image_bgr: BGR color tuple.
    :returns: RGB color tuple.
    """
    b, g, r = cv2.split(image_bgr)
    image_rgb = cv2.merge([r, g, b])
    return image_rgb


def pickle_save(file_path, obj):
    """
    Save an object to a given path as a pickle file.

    :param file_path: Destination file path.
    :param obj: Object to store at given path.
    :returns: None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(file_path):
    """
    Load an object from a given path to a pickle file.

    :param file_path: File path to the pickle file.
    :returns: The object that has been stored in the given file.
    """
    if not os.path.isfile(file_path):
        return None

    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
