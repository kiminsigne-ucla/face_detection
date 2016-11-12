from itertools import chain
import numpy as np

def determine_threshold(feature, scores, weights):
    """
    :param feature: Haar feature
    :param scores: two element list, first entry positive image scores, second entry negative image scores
    :param weights: two element list, first entry positive data weights, second entry negative data weights
    :return: updated feature with new threshold and parity based on positive and negative scores
    """

    # weight each score, np arrays give element-wise multiplication
    weighted_scores = [(np.array(scores[0]) * weights[0]), (np.array(scores[1]) * weights[1])]

    # for each weighted score, convert to tuple (score, 1) for positive samples and (score, -1) for negative samples
    weighted_scores[0] = [(score, 1) for score in weighted_scores[0]]
    weighted_scores[1] = [(score, -1) for score in weighted_scores[1]]
    # arrange all samples by weighted score
    samples = list(chain(*weighted_scores))
    samples.sort()

    cdef int pos_total, neg_total
    pos_total = len(weighted_scores[0])
    neg_total = len(weighted_scores[1])

    cdef int pos_below_thres, neg_below_thresh, error
    # establishing a threshold will split the sorted samples into two halves. We can try each partition, calculate how
    # many points are correctly classified, and choose the best threshold with the lowest error
    threshold_idx = None
    pos_below_thresh = 0
    neg_below_thresh = 0
    min_error = np.inf
    for i in range(len(samples)):
        if samples[i][1] == 1:
            pos_below_thresh += samples[i][0]
        else:
            neg_below_thresh += samples[i][0]
        error = min((pos_below_thresh + neg_total - neg_below_thresh),
                    (neg_below_thresh + pos_total - pos_below_thresh))
        if error < min_error:
            threshold_idx = i
            min_error = error

    # find x value at threshold index
    threshold = list(chain(*scores))[threshold_idx]

    pos_median = np.median([x[0] for x in weighted_scores[0]])
    neg_median = np.median([x[0] for x in weighted_scores[1]])
    if pos_median < neg_median:
        parity = 1
    else:
        parity = -1

    updated_feature = feature
    updated_feature.threshold = threshold
    updated_feature.parity = parity

    return updated_feature


def evaluate_classI(object self, integral_img):
    """
    :param integral_img: summed area table for image for fast calculation
    :return: self score for image
    """

    cdef double a, b, c, d, e, f, left_rect_sum, right_rect_sum, score

    # need six points to calculate sums for two adjacent rectangles
    a = integral_img[self.x][self.y]
    b = integral_img[self.x][self.y + self.width]
    c = integral_img[self.x][self.y + 2 * self.width]
    d = integral_img[self.x + self.height][self.y]
    e = integral_img[self.x + self.height][self.y + self.width]
    f = integral_img[self.x + self.height][self.y + 2 * self.width]
    left_rect_sum = e + a - b - d
    right_rect_sum = f + b - c - e
    if self.inverse == 0:  # white on left, dark on right
        score = right_rect_sum - left_rect_sum
    else:  # self.inverse == 1, dark on left, white on right
        score = left_rect_sum - right_rect_sum

    return score


def evaluate_classII(self, integral_img):
    """
    :param integral_img: summed area table for image for fast calculation
    :return: self score for image
    """

    cdef double a, b, c, d, e, f, left_rect_sum, right_rect_sum, score

    a = integral_img[self.x][self.y]
    b = integral_img[self.x][self.y + self.width]
    c = integral_img[self.x + self.height][self.y]
    d = integral_img[self.x + self.height][self.y + self.width]
    e = integral_img[self.x + 2 * self.height][self.y]
    f = integral_img[self.x + 2 * self.height][self.y + self.width]

    top_rect = d + a - b - c
    bottom_rect = f + c - d - e
    if self.inverse == 0:  # dark on top, white on bottom
        score = top_rect - bottom_rect
    else:  # self.inverse == 1, white on top, dark on bottom
        score = bottom_rect - top_rect

    return score


def evaluate_classIII(self, integral_img):
    """
    :param integral_img: summed area table for image for fast calculation
    :return: self score for image
    """

    cdef double a, b, c, d, e, f, g, h, left_rect_sum, right_rect_sum, score

    # need eight points to calculate sums for three adjacent rectangles
    a = integral_img[self.x][self.y]
    b = integral_img[self.x][self.y + self.width]
    c = integral_img[self.x][self.y + 2 * self.width]
    d = integral_img[self.x][self.y + 3 * self.width]
    e = integral_img[self.x + self.height][self.y]
    f = integral_img[self.x + self.height][self.y + self.width]
    g = integral_img[self.x + self.height][self.y + 2 * self.width]
    h = integral_img[self.x + self.height][self.y + 3 * self.width]

    left_rect = f + a - b - e
    middle_rect = g + b - c - f
    right_rect = h + c - d - g
    if self.inverse == 0:  # white-dark-white
        score = middle_rect - (left_rect + right_rect)
    else:  # self.inverse == 1, dark-white-dark
        score = (left_rect + right_rect) - middle_rect

    return score


def evaluate_classIV(self, integral_img):

    cdef double a, b, c, d, e, f, g, h, i, left_rect_sum, right_rect_sum, score

    a = integral_img[self.x][self.y]
    b = integral_img[self.x][self.y + self.width]
    c = integral_img[self.x][self.y + 2 * self.width]
    d = integral_img[self.x + self.height][self.y]
    e = integral_img[self.x + self.height][self.y + self.width]
    f = integral_img[self.x + self.height][self.y + 2 * self.width]
    g = integral_img[self.x + 2 * self.height][self.y]
    h = integral_img[self.x + 2 * self.height][self.y + self.width]
    i = integral_img[self.x + 2 * self.height][self.y + 2 * self.width]

    upper_left = e + a - b - d
    upper_right = f + b - c - e
    lower_left = h + d - e - g
    lower_right = i + e - f - h
    if self.inverse == 0:
        score = (upper_right + lower_left) - (upper_left + lower_right)
    else:
        score = (upper_left + lower_right) - (upper_right + lower_left)

    return score