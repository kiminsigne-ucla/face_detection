from realboost import run_classifier, realboost
from scipy.misc import imresize
from itertools import chain
from learn_faces import calculate_integral_imgs
from skimage.transform import integral_image
from functools import partial
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle


def detect_faces(image, classifier, min_scale, max_scale, step, negative_mining=False):
    """
    Use sliding window, at different scales, to detect face images
    :param image: Original image for face detection
    :param classifier: Real-boosted classifier
    :param min_scale: Minimum image scale
    :param max_scale: Maximum image scale
    :param step: Step in between image scales
    :param negative_mining: Return the ratio and scale for each detected face, necessary to extract images in hard
    negative mining
    :return: list of detected faces with coordinates (x1, y1, x2, y2) corrresponding to upper-left and lower-right
    corners, respectively. If negative mining, include ratio and scale, if not, include response of classifier (used in
    non-maximum suppression)
    """

    hard_negatives = []

    # for multiple image scales
    scales = np.arange(min_scale, max_scale, step)
    if negative_mining:
        # returns face_boxes and hard_negative images
        info = [detect_faces_mini(scale, image, classifier, negative_mining) for scale in scales]
        face_boxes = [x[0] for x in info]
        hard_negatives = [x[1] for x in info]
    else:
        face_boxes = [detect_faces_mini(scale, image, classifier, negative_mining) for scale in scales]

    # flatten list of lists
    face_boxes = list(chain(*face_boxes))
    hard_negatives = list(chain(*hard_negatives))

    if negative_mining:
        return face_boxes, hard_negatives
    else:
        return face_boxes


def detect_faces_mini(scale, image, classifier, negative_mining=False):

    # face_boxes = []
    # hard_negatives = []

    # scale image
    scaled_img = imresize(image, scale)
    # keep track of ratio between original width and new width
    ratio = image.shape[1] / float(scaled_img.shape[1])

    num_processes = 2
    pool = multiprocessing.Pool(processes=num_processes)

    # grab window coordinates
    window_coords = [[(i, j) for j in xrange(scaled_img.shape[1] - 16 + 1)] for i in xrange(scaled_img.shape[0] - 16 + 1)]
    window_coords = list(chain(*window_coords))
    func = partial(classify_window, scaled_img=scaled_img, classifier=classifier, ratio=ratio)
    face_boxes = pool.map(func, window_coords)
    # trim down None results
    face_boxes = [x for x in face_boxes if x is not None]
    if negative_mining:
        # grab images for hard negatives
        hard_negatives = extract_imgs(face_boxes, scaled_img, ratio)
        return face_boxes, hard_negatives
    else:
        return face_boxes


def classify_window(window_coord, scaled_img, classifier, ratio):

    i, j = window_coord
    window = scaled_img[i:i + 16, j:j + 16]
    # calculate integral image and pad
    window = np.pad(integral_image(window), pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)

    response = run_classifier(window, classifier, classify=False)
    if np.sign(response) == 1:
        # store box boundaries, upper left corner and lower right corner
        x1, y1 = int(i * ratio), int(j * ratio)
        x2, y2 = int((i + 16) * ratio), int((j + 16) * ratio)
        return x1, y1, x2, y2, response
    else:
        return None


def hard_negative_mining(images, classifier, weights, face_integral_imgs, nonface_integral_imgs,
                         min_scale, max_scale, step):
    """
    Run classifier on background images, any faces detected are hard negatives. Incorporate these into training as
    negative images, and re-train Realboost classifier for 100 more iterations (aka refine each feature) starting from
    weights from last training iteration

    :param images: Background images
    :param classifier: 100 features from Realboost
    :param weights: Weights from last training iteration of Realboost
    :param face_integral_imgs: Positive training images
    :param nonface_integral_imgs: Negative training images
    :param min_scale: Minimum image scale
    :param max_scale: Maximum image scale
    :param step: step beteween min_scale and max_scale
    :return: Classifier re-trained with hard negatives
    """

    # run face detector on background images, any identified faces are hard negatives
    print "Detecting hard negatives in background images..."

    neg_info = [detect_faces(image, classifier, min_scale, max_scale, step, negative_mining=True)
                             for image in images]
    pickle.dump(neg_info, open('neg_info.pkl', 'wb'))

    neg_boxes = [x[0] for x in neg_info]
    neg_images = [x[1] for x in neg_info]

    # neg_images = [extract_imgs(neg_boxes[i], images[i]) for i in range(len(images))]
    # flatten and convert to integral images
    hard_neg_integral_imgs = calculate_integral_imgs(chain(*neg_images))

    # need to incorporate hard negatives into training, so grab the highest negative sample weight and assign this to
    # each hard negative
    max_neg_weight = max(weights[1])
    # add these max weights to end of negative sample weights, add hard negative images to end of negative imgs
    weights[1] = np.append(weights[1], [max_neg_weight] * len(hard_neg_integral_imgs))
    # re-scale negative weights
    weights[1] /= sum(weights[1])

    new_nonface_integral_imgs = np.append(nonface_integral_imgs, hard_neg_integral_imgs, axis=0)
    print "Re-training Realboosted features for 100 iterations using hard negatives, starting with weights from " \
          "last Realboost training iteration"

    retrained = realboost(classifier, 100, face_integral_imgs, new_nonface_integral_imgs, weights)

    return retrained


def extract_imgs(boxes, scaled_image, ratio):
    """
    Return the image portions indicated by the box boundaries
    :param boxes: List of tuples, each entry is (x1, y1, x2, y2, response) corresponding to upper left corner
    and lower right corner and the classifier response. Coordinates are relative to original image scale
    :param scaled_image: Image to extract sub-images, must be at proper scale
    :param ratio: ratio between the original image coordinates and the scaled coordinates so that we can extract the
    proper 16x16 image
    :return: list of images within box coordinates
    """

    images = []
    for box in boxes:
        x1, y1, x2, y2, response = box
        scaled_x1, scaled_y1 = int(round(x1 / ratio)), int(round(y1 / ratio))
        scaled_x2, scaled_y2 = int(round(x2 / ratio)), int(round(y2 / ratio))
        img = scaled_image[scaled_x1:scaled_x2, scaled_y1:scaled_y2]
        images.append(img)

    return images


def nonmax_suppression(boxes, threshold):
    """
    If detected boxes overlap by more than threshold, choose box with highest score
    :param boxes: List of box coordinates, (x1, y1, x2, y2, response) where x1, y1 are upper left corner and x2, y2
    are lower-right corner, response is classifier score for that box (scale-dependent)
    :param threshold: Overlap threshold
    :return: Reduced set of non-overlapping boxes
    """

    selected_boxes = []

    if len(boxes) == 0:
        return selected_boxes

    # sort boxes by bottom right corner, with bottom-most corner first
    boxes_sorted = sorted(boxes, key=lambda x: (x[3], x[2]), reverse=True)
    # areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in boxes_sorted]

    while len(boxes_sorted) > 0:
        # starting with box in bottom-most right corner, first element of sorted list
        curr_box = boxes_sorted.pop(0)
        # initialize list of suppressed boxes
        suppress = [curr_box]
        # grab boxes that are within current box, lower-right corner >= top-left corner of current box
        within = []
        curr_x1, curr_y1 = curr_box[:2]
        for box in boxes_sorted:
            if box[2] > curr_x1 and box[3] > curr_y1:
                within.append(box)
            if box[2] <= curr_x1 and box[3] <= curr_y1:
                break

        # calculate area overlap between current box and boxes within region
        curr_area = (curr_box[2] - curr_box[0] + 1) * (curr_box[3] - curr_box[1] + 1)
        areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in within]
        overlaps = [float(curr_area) / float(area) for area in areas]
        # if over threshold, suppress all but one
        above_threshold = [within[i] for i in range(len(within)) if overlaps[i] >= threshold]
        if len(above_threshold) > 0:
            # choose box with highest score, score stored in last element of tuple
            max_value, max_index = max(((val, idx) for (idx, val) in enumerate(above_threshold)), key=lambda x:x[0][-1])
            selected_boxes.append(above_threshold[max_index])
            # suppress all images within current box that are above overlap threshold
            suppress.extend(above_threshold)
            # remove suppressed images from sorted boxes
            boxes_sorted = [box for box in boxes_sorted if box not in suppress]
        else:
            selected_boxes.append(curr_box)

    return selected_boxes


def graph_boxes_on_image(boxes, image, filename='boxed_faces.png'):
    """
    Draw red rectangles on image according to box coordinates
    :param boxes: list of box coordinates, (x1, y1, x2, y2) which are upper-left and lower-right corners, respectively
    :param image: image to draw boxes on
    :param filename: name for saved plot
    :return: None, saves plot
    """

    fig= plt.figure(figsize=(14, 8), dpi=1000)
    ax = fig.add_subplot(111)
    plt.axis('off')

    # display image
    ax.imshow(image)

    # Create a Rectangle patch, first argument (x,y) which is upper-left corner, followed by width, height
    # matrix coordinates are opposite of image axes
    for rect in [patches.Rectangle((box[1], box[0]), width=box[3] - box[1], height=box[2] - box[0],
                                   linewidth=1, edgecolor='r', facecolor='none') for box in boxes]: ax.add_patch(rect)
    fig.savefig(filename)