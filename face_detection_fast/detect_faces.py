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
    num_processes = 30
    pool = multiprocessing.Pool(processes=num_processes)
    func = partial(detect_faces_mini, image=image, classifier=classifier, negative_mining=negative_mining)

    # for multiple image scales
    scales = np.arange(min_scale, max_scale, step)
    if negative_mining:
        face_boxes, hard_negatives = pool.map(func, scales)
    else:
        face_boxes = pool.map(func, scales)

    # flatten list of lists
    face_boxes = list(chain(*face_boxes))
    hard_negatives = list(chain(*hard_negatives))

    if negative_mining:
        return face_boxes, hard_negatives
    else:
        return face_boxes


def detect_faces_mini(scale, image, classifier, negative_mining=False):

    face_boxes = []
    hard_negatives = []

    # scale image
    scaled_img = imresize(image, scale)
    # keep track of ratio between original width and new width
    ratio = image.shape[1] / float(scaled_img.shape[1])
    for i in xrange(scaled_img.shape[0] - 16 + 1):
        for j in xrange(scaled_img.shape[1] - 16 + 1):
            window = scaled_img[i:i + 16, j:j + 16]
            # calculate integral image and pad
            window = np.pad(integral_image(window), pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)
            response = run_classifier(window, classifier, classify=False)
            if np.sign(response) == 1:
                # store box boundaries, upper left corner and lower right corner
                x1, y1 = int(i * ratio), int(j * ratio)
                x2, y2 = int((i + 16) * ratio), int((j + 16) * ratio)
                if negative_mining:
                    face_boxes.append((x1, y1, x2, y2, ratio, scale))
                    hard_negatives.append(scaled_img[i:i + 16, j:j + 16])
                else:
                    face_boxes.append((x1, y1, x2, y2, response))

    if negative_mining:
        return face_boxes, hard_negatives
    else:
        return face_boxes


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

    neg_boxes, neg_images = [detect_faces(image, classifier, min_scale, max_scale, step, negative_mining=True)
                             for image in images]
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


def extract_imgs(boxes, image):
    """
    Return the image portions indicated by the box boundaries
    :param boxes: List of tuples, each entry is (x1, y1, x2, y2, ratio) corresponding to upper left corner
    and lower right corner and the ratio between the original image coordinates and the scaled coordinates so that we
    can extract the proper 16x16 image
    :param image: Image to extract sub-images
    :return: list of images within box coordinates
    """

    images = []
    for box in boxes:
        x1, y1, x2, y2, ratio, scale = box
        scaled_x1, scaled_y1 = int(round(x1 / ratio)), int(round(y1 / ratio))
        scaled_x2, scaled_y2 = int(round(x2 / ratio)), int(round(y2 / ratio))
        scaled_img = imresize(image, scale)
        img = scaled_img[scaled_x1:scaled_x2, scaled_y1:scaled_y2]
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
    boxes_sorted = sorted(boxes, key=lambda x: (x[2], x[3]), reverse=True)
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
        if len(above_threshold) >= 0:
            # choose box with highest score, score stored in last element of tuple
            max_index, max_value = max(enumerate(above_threshold), key=lambda x: x[-1])
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

    fig, ax = plt.subplots(1)

    # display image
    ax.imshow(image)

    # Create a Rectangle patch, first argument (x,y) which is upper-left corner, followed by width, height
    for rect in [patches.Rectangle((box[0], box[1]), width=box[3] - box[1], height=box[2] - box[0],
                                   linewidth=1, edgecolor='r', facecolor='none') for box in boxes]: ax.add_patch(rect)
    fig.savefig(filename)
