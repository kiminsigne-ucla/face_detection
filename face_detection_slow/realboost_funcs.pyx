import numpy as np
from math import log, sqrt
from copy import copy
from itertools import chain


def calculate_bins(feature, face_integral_imgs, nonface_integral_imgs, weights, total_weight=False):

    # This function exactly the same as in weighted error, but useful when you only have to calculate it for one feature
    # like when updating weights

    cdef double pos_bin_weights, neg_bin_weights, bin_weight

    scores = [[feature.evaluate(img) for img in face_integral_imgs],
              [feature.evaluate(img) for img in nonface_integral_imgs]]

    # weight each score, np arrays give element-wise multiplication
    weighted_scores = [(np.array(scores[0]) * weights[0]), (np.array(scores[1]) * weights[1])]

    # for each score, convert to tuple (score, 1, weight) for positive samples and
    # (score, -1, weight) for negative samples
    weighted_scores[0] = [(weighted_scores[0][i], 1, weights[0][i]) for i in xrange(len(weighted_scores[0]))]
    weighted_scores[1] = [(weighted_scores[1][i], -1, weights[1][i]) for i in xrange(len(weighted_scores[1]))]

    # arrange all samples by score
    samples = list(chain(*weighted_scores))
    samples.sort()

    # calculate b bins based on score range
    hist = np.histogram([x[0] for x in samples], feature.weight)
    # don't care about the first bin
    bins = hist[1][1:]

    # use small pseudo-count so we don't have to use infinity
    pseudo_count = 1/float(len(scores[0])+len(scores[1])) ** 2

    bin_boundaries = copy(bins)
    bin_weights = []
    bin_weights_total = []

    sample_count = 0
    for bin in hist[0]:  # contains number of values in each bin
        bin_samples = samples[sample_count:sample_count+bin]
        sample_count += bin
        neg_bin_weights = 0
        pos_bin_weights = 0
        for x in bin_samples:
            if x[1] == 1:
                pos_bin_weights += x[2]
            else:
                neg_bin_weights += x[2]
        if pos_bin_weights == 0 and neg_bin_weights == 0:
            bin_weight = 0
        # else:
        #     if pos_bin_weights == 0:
        #         pos_bin_weights = pseudo_count
        #     if neg_bin_weights == 0:
        #         neg_bin_weights = pseudo_count
        # all negative samples, give low pseudo-count to positive
        elif pos_bin_weights == 0 and neg_bin_weights != 0:
            bin_weight = 0.5 * log(pseudo_count / neg_bin_weights)
        # all positive samples, low pseudo-count to negatives
        elif pos_bin_weights != 0 and neg_bin_weights == 0:
            bin_weight = 0.5 * log(pos_bin_weights / pseudo_count)
        else:
            bin_weight = 0.5 * log(pos_bin_weights / neg_bin_weights)

        bin_weights_total.append(bin_weight)
        bin_weights.append((pos_bin_weights, neg_bin_weights))

    if total_weight:
        return bin_boundaries, bin_weights_total
    else:
        return bin_boundaries, bin_weights


def calculate_weighted_error(feature, face_integral_imgs, nonface_integral_imgs, weights):
    """
    Calculate weighted error. We choose weak classifier to minimize Z = 2 * (for all B) sqrt( pt(b) * qt(b) )
    where pt(b) and qt(b) are sum of positive and negative weights, respectively, for all samples in bin b

    return: feature error, Z
    """

    cdef double error
    bin_boundaries, bin_weights = calculate_bins(feature, face_integral_imgs, nonface_integral_imgs, weights,
                                                 total_weight=False)
    # each bin weight is tuple (positive, negative)
    error = 2 * sum([sqrt(x[0] * x[1]) for x in bin_weights])

    return error
