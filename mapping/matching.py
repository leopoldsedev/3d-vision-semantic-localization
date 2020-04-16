import images


class FeatureMatch():

    def __init__(self, image_idx1, detection_idx1, image_idx2, detection_idx2):
        self.image_idx1 = image_idx1
        self.detection_idx1 = detection_idx1
        self.image_idx2 = image_idx2
        self.detection_idx2 = detection_idx2

    def __repr__(self):
        return f'FeatureMatch(image_idx1={self.image_idx1}, detection_idx1={self.detection_idx1}, image_idx2={self.image_idx2}, detection_idx2={self.detection_idx2})'


"""
Matches the given detections between images

:param image_paths: The list of image paths (in order of the image sequence).
:param detections: The detection dictionary from detection.detect_traffic_signs().
:returns: List of instances of FeatureMatch
"""
def match_detections(image_dir_path, detections):
    # TODO Properly implement matching
    result = []

    image_paths = images.get_image_path_list(image_dir_path)

    # TODO Undo
    for i in range(len(image_paths)-1):
        result.append(FeatureMatch(i, 0, i+1, 0))
        result.append(FeatureMatch(i, 1, i+1, 1))

    return result
