import glob
from os.path import join, basename


def get_image_path_list(image_dir_path):
    return glob.glob(join(image_dir_path, '*.jpg'))#[50:-50] # TODO Undo

def get_image_names(image_dir_path):
    image_paths = get_image_path_list(image_dir_path)
    image_names = list([basename(path) for path in image_paths])
    return image_names
