import glob
import numpy as np
from os.path import join, basename

import util


def get_pickle_files(dir_path):
    return sorted(glob.glob(join(dir_path, '*.pickle')))


DATASET_NAME = '10_right'
DIR_PATH = join('./output/scores/', DATASET_NAME)
OUTPUT_PATH = join('./output/scores/merged', DATASET_NAME + '.pickle')

pickle_files = get_pickle_files(DIR_PATH)

result = {}

for pickle_file in pickle_files:
    image_name = basename(pickle_file)[:-len('.pickle')]
    scores = util.pickle_load(pickle_file)
    result[image_name] = scores

#print(result[list(result.keys())[0]].shape)

util.pickle_save(OUTPUT_PATH, result)
