from enum import Enum
import os
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
from sklearn.preprocessing import StandardScaler


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


# Templates from https://commons.wikimedia.org/wiki/Road_signs_of_Spain
#template_name = 'crossing'
template_name = 'yield'
#template_name = 'roundabout'

output_path = './template_matching_output'
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

templates_path = './templates/ideal'
templates_images_path = join(templates_path, 'images')
templates_masks_path = join(templates_path, 'masks')
template_image_path = join(templates_images_path, template_name + '.png')
template_mask_path = join(templates_masks_path, template_name + '.png')

template = cv2.imread(template_image_path)
template_mask = cv2.imread(template_mask_path)

#template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#template = cv2.Canny(template, 50, 200)
#b,g,r = cv2.split(template)       # get b,g,r
#template_rgb = cv2.merge([r,g,b])
#plt.imshow(template_mask)
#plt.show()
(tH, tW) = template.shape[:2]

"""
Detects traffic in the image at the given path

:param image_path: The path of the image
:returns: List of instances of TrafficSignDetection
"""
maxVals = []
def detect_traffic_signs_in_image(image_path):
    def kmeans_clustering(image):
        Z = image.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 32
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        return res2

    def highlight_colors(image):
        print("highlight_colors")
        chan0 = image[:,:,0]
        chan1 = image[:,:,1]
        chan2 = image[:,:,2]

        result = image
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #print(hsv)
        # Saturation to max
        #hsv[:,:,1] = hsv[:,:,1] * 2
        # Lightness to mean
        #hsv[:,:,2] = 255/2
        #result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize= (8,8))
        #equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256
        chan0_clahe = clahe.apply(chan0)
        chan1_clahe = clahe.apply(chan1)
        chan2_clahe = clahe.apply(chan2)
        merged = np.array([chan0_clahe, chan1_clahe, chan2_clahe])
        result = np.swapaxes(np.swapaxes(merged, 0, 1), 1, 2)

        #print(clahe.getTilesGridSize())
        #result = cv2.hconcat([image, result])
        return result

    def gammaCorrection(image, gamma):
        ## [changing-contrast-brightness-gamma-correction]
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        res = cv2.LUT(image, lookUpTable)
        ## [changing-contrast-brightness-gamma-correction]

        img_gamma_corrected = res #cv2.hconcat([image, res])
        return img_gamma_corrected

    results = []
    image = cv2.imread(image_path)
    alpha = 1.3 # Contrast control (1.0-3.0)
    beta = 40 # Brightness control (0-100)

    adjusted = image[300:,:]
    #adjusted = cv2.GaussianBlur(adjusted, (3,3), 5)
    #adjusted = gammaCorrection(adjusted, 0.5)
    #adjusted = cv2.convertScaleAbs(adjusted, alpha=alpha, beta=beta)
    adjusted = cv2.medianBlur(adjusted, 3)
    #adjusted = highlight_colors(adjusted)
    #adjusted = kmeans_clustering(adjusted)
    #d = 31
    #adjusted = cv2.bilateralFilter(adjusted, d, d*2, d/2)
    #cv2.imwrite(basename(image_path), adjusted)
    #adjusted = cv2.hconcat([image[300:,:], adjusted])
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    found = None
    print(image_path)
    b,g,r = cv2.split(adjusted)       # get b,g,r
    adjusted_rgb = cv2.merge([r,g,b])
    #plt.imshow(adjusted_rgb)
    #plt.show()

    # TODO Filter detections:
    # - Too high variance of matchTemplate score -> discard
    # - Max has to be higher than 0.3
    # - Always examine only scores that are at least 90% of the maximum
    # - Not consistent over a few resizes of the template -> discard

    fig = plt.figure(figsize = (18,8))
    timer = fig.canvas.new_timer(interval = 1000)
    def close_event():
        plt.close()
    timer.add_callback(close_event)

    # loop over the scales of the image
    for scale in np.linspace(0.3, 0.1, 30):
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        width = int(template.shape[1]*scale)
        height = int(template.shape[0]*scale)
        dim = (width,height)
        resized = cv2.resize(template, dim, interpolation = cv2.INTER_CUBIC)
        resized_mask = cv2.resize(template_mask, dim, interpolation = cv2.INTER_NEAREST)
        #r = template.shape[1] / float(resized.shape[1])
        print(gray.shape)
        print(resized.shape)
        print(resized_mask.shape)

        # if the resized image is smaller than the template, then break
        # from the loop
        #if resized.shape[0] < tH or resized.shape[1] < tW:
                #break

        # detect edges and apply template
        #edged = cv2.Canny(resized, 50, 200)
        #plt.imshow(gray, cmap='gray')
        #plt.show()

        #plt.imshow(resized, cmap='gray')
        #plt.show()
        result = cv2.matchTemplate(adjusted, resized, cv2.TM_CCORR_NORMED, mask=resized_mask)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        print('result')
        print(result.shape)
        print(f'min={np.min(result)}')
        print(f'max={np.max(result)}')
        print(f'mean={np.mean(result)}')
        print(f'var={np.var(result)}')
        #plt.figure(figsize = (18,8))
        #filt = result
        #filt[filt < 0.7] = 0
        #plt.imshow(filt)
        #plt.show()

        #print(maxLoc)
        #print(maxVal)
        #flat = np.reshape(result, (1, result.size))[0,:]
        #print(flat.shape)
        #plt.hist(flat, bins=100)
        #plt.boxplot(flat)
        #plt.show()

        max_val = np.max(result)
        max_val = maxVal
        def argmax2d(arr):
            return np.unravel_index(np.argmax(arr), arr.shape)

        result1d = np.reshape(result, (result.size,))
        high_enough = np.where(result1d >= 0.9)[0]
        #high_enough = np.argpartition(result1d, -1)[-1:]
        print("##############")
        print(high_enough.size)
        for r in high_enough:
            s = result1d[r]
            loc = np.unravel_index(r, result.shape)
            loc = maxLoc
            #print("###########")
            #print(r)
            #print(s)
            #print(loc)
            x = loc[0] + width / 2
            y = loc[1] + height / 2
            plt.scatter(x, y, color='red')
            results.append((x, y, width, height, s))

    if len(results) > 0:
        x, y, temp_w, temp_h, score = zip(*results)
        points = np.vstack([x,y]).T
        scores = np.array(score)
        temp_w = np.array(temp_w)
        temp_h = np.array(temp_h)

        points_normalized = StandardScaler().fit_transform(points.astype(float))

        # Clustering
        db_scan = DBSCAN(eps=0.3, min_samples=1)
        db_scan_result = db_scan.fit(points_normalized)
        labels = db_scan_result.labels_

        unique_labels = set(labels)
        print(unique_labels)

        clusters = []
        cluster_scores = []
        cluster_width = []
        cluster_height = []
        # Process results
        for label in unique_labels:
            # Do not include noise in the result
            if label == -1:
                continue

            cluster_mask = (labels == label)
            cluster_points = points[cluster_mask]
            cs = scores[cluster_mask]
            cw = temp_w[cluster_mask]
            ch = temp_h[cluster_mask]
            #print(cluster_points)
            #print(cs)
            assert(cs.shape[0] == cluster_points.shape[0])
            clusters.append(cluster_points[np.argmax(cs)])
            cluster_scores.append(np.max(cs))
            cluster_width.append(np.max(cw))
            cluster_height.append(np.max(ch))

        print(clusters)
        #plt.plot(x, y, 'o')
        scales = (5*np.array(cluster_scores))**4
        #print(scales)

        for c, w, h in zip(clusters, cluster_width, cluster_height):
            pt1 = (int(c[0]-w/2),int(c[1]-h/2))
            pt2 = (int(c[0]+w/2),int(c[1]+h/2))
            cv2.rectangle(adjusted, pt1, pt2, (0,0,0), 2)

    cv2.imwrite(join(output_path, basename(image_path)), adjusted)
    b,g,r = cv2.split(adjusted)       # get b,g,r
    adjusted_rgb = cv2.merge([r,g,b])
    #plt.imshow(adjusted_rgb)
    #print(results)
    #timer.start()
    #plt.show()

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

    for image_path in image_paths[250::1]:
        image_name = basename(image_path)
        result[image_name] = detect_traffic_signs_in_image(image_path)

    return result

if __name__ == '__main__':
    #path = '/home/patricia/3D/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images'
#    path = '/home/patricia/3D/multiscale-template-matching/multiscale-template-matching/malaga/testall'
    #path = './images-10'
    path = './07/images/rectified'

    # save results in a txt file
    #f = open("yield.txt","a+")
    results = detect_traffic_signs(path)
    #for i,j in results.items():

        #f.write("%s %s \n" % (i,j))
    #f.close()
    #plt.hist(maxVals, density = True, bins = 1000)
    #plt.show()
