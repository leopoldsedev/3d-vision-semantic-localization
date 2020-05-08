from enum import Enum
import os
import pickle
import shutil
from os.path import basename, join
import images
#for detection
import numpy as np
import imutils
import cv2
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, minmax_scale
from collections import namedtuple


TrafficSignDetection = namedtuple('TrafficSignDetection', ['x', 'y', 'width', 'height', 'sign_type', 'score'])


class TrafficSignType(Enum):
    CROSSING = 1
    ROUNDABOUT = 2
    YIELD = 3
    # TODO Add other types

ALL_SIGN_TYPES = {TrafficSignType.CROSSING, TrafficSignType.YIELD, TrafficSignType.ROUNDABOUT}

sign_type_colors = {
    TrafficSignType.CROSSING: (255, 0, 0),
    TrafficSignType.YIELD: (0, 0, 255),
    TrafficSignType.ROUNDABOUT: (0, 255, 0),
}


def kmeans_clustering(image, K):
    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2


def gamma_correction(image, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    img_gamma_corrected = cv2.LUT(image, lookUpTable)

    return img_gamma_corrected


def clahe_correction(image):
    chan0 = image[:,:,0]
    chan1 = image[:,:,1]
    chan2 = image[:,:,2]

    result = image
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    chan0_clahe = clahe.apply(chan0)
    chan1_clahe = clahe.apply(chan1)
    chan2_clahe = clahe.apply(chan2)
    merged = np.array([chan0_clahe, chan1_clahe, chan2_clahe])
    result = np.swapaxes(np.swapaxes(merged, 0, 1), 1, 2)

    return result


def covariance_metric(cov):
    # The determinant is a good metric for the 'spread' of a covariance matrix. See https://stats.stackexchange.com/a/63037
    # Alternative: Multiply the roots of the eigenvalues to calculate the 'area of the spread'
    return np.sqrt(np.linalg.det(cov))


def preprocess_image(image):
    alpha = 1.8 # Contrast control (1.0-3.0)
    beta = 100 # Brightness control (0-100)

    result = image.copy()
    #result = cv2.GaussianBlur(result, (3,3), 5)
    #result = gamma_correction(result, 0.4)
    #result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    #result = clahe_correction(result)
    #result = cv2.medianBlur(result, 3)
    #result = kmeans_clustering(result, 32)
    #d = 31
    #result = cv2.bilateralFilter(result, d, d*2, d/2)
    return result


def detect_template_resize(image, template, template_mask, sign_type, grayscale):
    assert(template.shape == template_mask.shape)

    preprocessed = preprocess_image(image)

#     comparison = cv2.hconcat([image, preprocessed])
#     plt.figure(figsize = (18,8))
#     show_image_bgr(comparison)

    if grayscale:
        preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    min_height = 20
    max_height = 100
    height_steps = range(min_height, max_height + 1, 1)
    top_n = 10

    raw_matches = []

    for height in height_steps:
        template_scale = height / template.shape[0]
        width = int(template.shape[1] * template_scale)
        dim = (width, height)

        resized = cv2.resize(template, dim, interpolation = cv2.INTER_CUBIC)
        resized_mask = None

        if template_mask is not None:
            resized_mask = cv2.resize(template_mask, dim, interpolation = cv2.INTER_NEAREST)
            #show_image_bgr(resized)
            #show_image_bgr(resized_mask)

        result = cv2.matchTemplate(preprocessed, resized, cv2.TM_CCORR_NORMED, mask=resized_mask)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

#         plt.figure(figsize = (18,8))
#         plt.scatter(260, 120, s=1, color='red')
#         plt.scatter(maxLoc[0], maxLoc[1], s=1, color='red')
#         filt = result
#         filt[filt < 0.8] = 0.8
#         plt.imshow(result)
#         plt.show()

        result1d = np.reshape(result, (result.size,))

        score_threshold = 0.70
        scores_over_threshold = result1d[result1d >= score_threshold]
        if (len(scores_over_threshold) > 0):
            print(f'dem={dim} -- Found {len(scores_over_threshold)} points over theshold (max={np.max(result1d):.4f}).')
            # Strech scores over threshold to interval [0.0, 1.0]
            #scores_over_threshold = np.interp(scores_over_threshold, [score_threshold, 1.0], [0.0,1.0])
            print(f'min={np.min(scores_over_threshold)}')
            print(f'max={np.max(scores_over_threshold)}')
            print(f'mean={np.mean(scores_over_threshold)}')
            print(f'median={np.median(scores_over_threshold)}')
            print(f'var={np.var(scores_over_threshold)}')
            #plt.hist(scores_over_threshold, bins=[0.0, 0.5, 0.7, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0])
    #         plt.hist(scores_over_threshold, bins=100)
    #         plt.show()

            top_match_indices = np.argpartition(result1d, -top_n)[-top_n:]
            #top_match_indices = [np.argmax(result1d)]
            for idx in top_match_indices:
                score = result1d[idx]
                if score >= score_threshold:
                    y, x = np.unravel_index(idx, result.shape)
                    center_x = x + width / 2
                    center_y = y + height / 2
                    raw_matches.append((center_x, center_y, width, height, score))

    print(f'{len(raw_matches)} top matches.')

    result = []
    if len(raw_matches) > 0:
        x, y, temp_w, temp_h, score = zip(*raw_matches)
        points = np.vstack([x,y]).T
        scores = np.array(score)
        temp_w = np.array(temp_w)
        temp_h = np.array(temp_h)

        sizes = (10*np.interp(scores, [np.min(scores), 1.0], [0.01,1.0]))**8
        plt.scatter(x, y, s=sizes, color='red')

        # TODO Maybe normalize pixel coordinates before clustering so that the
        # chosen eps is invariant to the image size.

        # min_samples depends on the number of sizes of the template that were matched and the number of top matches that are considered for each size
        db_scan = DBSCAN(eps=5, min_samples=len(raw_matches)*0.005)
        db_scan_result = db_scan.fit(points)
        labels = db_scan_result.labels_

        unique_labels = set(labels)
        print(unique_labels)

        clusters = []
        cluster_points = []
        cluster_scores = []
        cluster_width = []
        cluster_height = []
        # Process results
        for label in unique_labels:
            # Do not include noise in the result
            if label == -1:
                continue

            cluster_mask = (labels == label)
            cp = points[cluster_mask]
            cs = scores[cluster_mask]
            cw = temp_w[cluster_mask]
            ch = temp_h[cluster_mask]

            assert(cs.shape[0] == cp.shape[0])

            highest_score_idx = np.argmax(cs)
            clusters.append(cp[highest_score_idx])
            cluster_points.append(cp)
            cluster_scores.append(cs)
            cluster_width.append(cw[highest_score_idx])
            cluster_height.append(ch[highest_score_idx])

        print(clusters)

        for c, p, w, h, s in zip(clusters, cluster_points, cluster_width, cluster_height, cluster_scores):
            mean = np.mean(p, axis=0)
            unbiased = p - mean
            weight = (np.interp(np.max(s), [np.min(scores), 1.0], [0.0, 1.0]))**8
            #print(weight)
            cov = np.cov(unbiased.T)
            # TODO Handle case where all points are on a perfectly straight line (cov == 0 in that case) better by calculating the covariance metric differently in that case
            cov_metric = covariance_metric(cov)
            if cov_metric == 0:
                print('Covariance matrix is 0, skipping...')
                continue

            metric = 100000 * weight * 1 / cov_metric

            print(f'mean={mean}')
            print(f'cov_metric={cov_metric}')
            print(f'weight={weight}')
            print(f'metric={metric}')

            if metric < 1:
                continue

            #print(cov)
            x = c[0]
            y = c[1]
            detection = TrafficSignDetection(x=x, y=y, width=w, height=h, sign_type=sign_type, score=np.max(s))

            pt1 = (int(x-w/2),int(y-h/2))
            pt2 = (int(x+w/2),int(y+h/2))
            thickness = int((10*np.interp(np.max(s), [np.min(scores),np.max(scores)], [0.01,1.0])))
            color = (0, 0, 255)
            cv2.rectangle(preprocessed, pt1, pt2, color, thickness)

            result.append(detection)

    #plt.imshow(bgr_to_rgb(preprocessed))
    #timer.start()
    #plt.show()

    return result


def detect_traffic_signs_by_template(image, sign_types):
    horizon_cutoff = 300
    cutoff = image[horizon_cutoff:,:]

    # Templates were taken from https://commons.wikimedia.org/wiki/Road_signs_of_Spain
    templates_path = './templates/ideal'
    templates_images_path = join(templates_path, 'images')
    templates_masks_path = join(templates_path, 'masks')

    def sign_type_to_paths(sign_type):
        extension = '.png'
        file_name = sign_type.name.lower() + extension
        template_image_path = join(templates_images_path, file_name)
        template_mask_path = join(templates_masks_path, file_name)
        return (sign_type, template_image_path, template_mask_path)

    template_paths = map(sign_type_to_paths, sign_types)

    detections = []
    for sign_type, template_image_path, template_mask_path in template_paths:
        if not os.path.isfile(template_image_path):
            print(f'Template image for sign type \'{sign_type.name}\' does not exist at path \'{template_image_path}\'')
            continue

        template = cv2.imread(template_image_path)
        template_mask = None

        # Masks are optional
        if os.path.isfile(template_mask_path):
            template_mask = cv2.imread(template_mask_path)

        print(f'Detecting signs of type \'{sign_type.name}\'...')
        template_detections = detect_template_resize(cutoff, template, template_mask, sign_type, False)

        print(f'Found {len(template_detections)} signs of type \'{sign_type.name}\'.')
        detections.extend(template_detections)

    def correct_detection(detection):
        cutoff_y = detection.y
        corrected_y = cutoff_y + horizon_cutoff
        return detection._replace(y=corrected_y)

    corrected_detections = list(map(correct_detection, detections))

    return corrected_detections


def bgr_to_rgb(image_bgr):
    b, g, r = cv2.split(image_bgr)
    image_rgb = cv2.merge([r, g, b])
    return image_rgb

def show_image_bgr(image_bgr):
    image_rgb = bgr_to_rgb(image_bgr)
    plt.imshow(image_rgb)
    plt.show()


def show_image_gray(image_gray):
    plt.imshow(image_gray, cmap='gray')
    plt.show()


def draw_detection_in_image(image, detection):
    x = detection.x
    y = detection.y
    w = detection.width
    h = detection.height
    s = detection.score

    pt1 = (int(x-w/2),int(y-h/2))
    pt2 = (int(x+w/2),int(y+h/2))
    thickness = 2
    color = sign_type_colors[detection.sign_type]
    cv2.rectangle(image, pt1, pt2, color, thickness)


"""
Detects traffic in the image at the given path

:param image_path: The path of the image
:returns: List of instances of TrafficSignDetection
"""
def detect_traffic_signs_in_image(image, sign_types):
    return detect_traffic_signs_by_template(image, sign_types)


"""
Detects traffic in the images at the given paths

:param image_paths: A list of paths to images
:returns: A dictionary where the keys are image paths and the values are list of instances of TrafficSignDetection
"""
def detect_traffic_signs(image_dir_path):
    result = {}

    debug_output_path = './output/template_matching_output'
    if os.path.exists(debug_output_path):
        shutil.rmtree(debug_output_path)
    os.makedirs(debug_output_path)

    image_paths = images.get_image_path_list(image_dir_path)
    image_count = len(image_paths)
    print('Processing {} images.'.format(image_count))

    cooldowns = dict.fromkeys(ALL_SIGN_TYPES, 0)
    lookup_attempts = dict.fromkeys(ALL_SIGN_TYPES, 0)

    # Interesting indices: 250, 280, 330, 380
    for image_path in image_paths[0:1000:1]:
        print(image_path)

        image_name = basename(image_path)
        image = cv2.imread(image_path)

        for sign_type in ALL_SIGN_TYPES:
            cooldowns[sign_type] = max(0, cooldowns[sign_type] - 1)

        # Only detect sign types where the cooldown is 0
        sign_types = set([sign_type for sign_type in ALL_SIGN_TYPES if cooldowns[sign_type] == 0])
        print(f'Looking for sign types: {sign_types}')

        image_detections = detect_traffic_signs_in_image(image, sign_types)
        result[image_name] = image_detections

        # Get all sign types that where detected in the current image
        detected_sign_types = set((detection.sign_type for detection in image_detections))

        # Get all sign types that where NOT detected in the current image, but where being searched for
        not_detected_sign_types = sign_types - detected_sign_types

        # Increase cooldowns for all signs that were not detected to improve performance
        for sign_type in detected_sign_types:
            lookup_attempts[sign_type] = 0

        for sign_type in not_detected_sign_types:
            lookup_attempts[sign_type] += 1
            cooldowns[sign_type] = min(11, 2**(lookup_attempts[sign_type] - 1))

        # Save debug image
        for detection in image_detections:
            draw_detection_in_image(image, detection)

        image_debug_path = join(debug_output_path, image_name)
        cv2.imwrite(image_debug_path, image)

    return result


def save_detections(file_path, detections):
    with open(file_path, 'wb') as f:
        pickle.dump(detections, f)


def load_detections(file_path):
    if not os.path.isfile(file_path):
        return None

    with open(file_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    #path = '/home/patricia/3D/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images'
#    path = '/home/patricia/3D/multiscale-template-matching/multiscale-template-matching/malaga/testall'
    #path = './images-10'
    # path = './07/images/rectified'
    path = '/Users/sahanpaliskara/Documents/3dVision/malaga-urban-dataset-extract-07/images'

    # save results in a txt file
    f = open("yield.txt","a+")
    results = detect_traffic_signs(path)
    for i,j in results.items():

        f.write("%s %s \n" % (i,j))
    f.close()
    plt.show()