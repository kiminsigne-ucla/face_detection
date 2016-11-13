from functools import partial
from realboost_funcs import calculate_bins, calculate_weighted_error
import numpy as np
import multiprocessing
import adaboost
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time


def realboost(features, num_iterations, face_integral_imgs, nonface_integral_imgs, weights=None):
    """

    :param features: Weak features for boosting
    :param num_iterations: Number of training iterations
    :param face_integral_imgs: Positive training images
    :param nonface_integral_imgs: Negative training images
    :param weights: Set of initial weights (optional)
    :return: Real-boosted features
    """

    num_processes = 2

    # reset each feature's weight/alpha field to 50, this attribute will now be number of bins used to approximate
    # classifier function
    for x in features:
        x.weight = 64

    m = len(face_integral_imgs) + len(nonface_integral_imgs)
    if not weights:
        initial_weight = 1 / float(m)
        weights = [[np.array([initial_weight] * len(face_integral_imgs)),
                    np.array([initial_weight] * len(nonface_integral_imgs))]]

    boosted_features = []

    pool = multiprocessing.Pool(processes=num_processes)

    for t in range(num_iterations):
        print "Iteration", t
        start = time.time()
        func = partial(calculate_weighted_error, face_integral_imgs=face_integral_imgs,
                       nonface_integral_imgs=nonface_integral_imgs, weights=weights[t])
        errors = pool.map(func, features)
        # choose classifier with lowest weighted error
        min_error, min_idx = min((val, idx) for (idx, val) in enumerate(errors))

        feat_new = features.pop(min_idx)
        boosted_features.append(feat_new)

        # update weights for each data point
        new_weights, feat_new_bin_info = update_weights(feat_new, face_integral_imgs, nonface_integral_imgs,
                                                           weights[t], pool)
        weights.append(new_weights)
        # store tuple of bin boundaries and bin weights for selected feature, needed for final classifier
        feat_new.weight = feat_new_bin_info

        # save progress for debugging
        pickle.dump(weights[t+1], open('realboost_tmp_weights.pkl', 'wb'))
        pickle.dump(boosted_features, open('real_boosted_features_tmp.pkl', 'wb'))

        # update feature thresholds based on new weights
        func = partial(adaboost.update_threshold, face_integral_imgs=face_integral_imgs,
                       nonface_integral_imgs=nonface_integral_imgs, weights=weights[t + 1])
        features = pool.map(func, features)
        end = time.time()
        print end - start

    # save weights of last iteration to re-train with hard negatives
    pickle.dump(weights[-1], open('realboost_weights.pkl', 'wb'))

    return boosted_features


def update_weights(feature, face_integral_imgs, nonface_integral_imgs, weights, pool):
    """
    new_weight = old_weight * exp(-yi * ht,b(xi))

    :param feature: newly added classifier
    :param face_integral_imgs: positive integral images
    :param nonface_integral_imgs: negative integral images
    :param weights: all weights, ith entry are weights at ith iteration. Each entry is two element list where first
    entry is positive weights and second entry is negative weights
    :param pool: multiprocessing pool
    :return: updated weights and bin weights for selected feature
    """

    new_weights = [[], []]

    bin_boundaries, bin_weights = calculate_bins(feature, face_integral_imgs, nonface_integral_imgs, weights,
                                                 total_weight=True)

    func = partial(calculate_single_bin_score, feature=feature, bin_weights=bin_weights, bin_boundaries=bin_boundaries)
    score_bin_weights = pool.map(func, face_integral_imgs)
    # multiple by -yi
    score_bin_weights = [-x for x in score_bin_weights]
    new_weights[0] = weights[0] * np.array(np.exp(score_bin_weights))

    func = partial(calculate_single_bin_score, feature=feature, bin_weights=bin_weights, bin_boundaries=bin_boundaries)
    score_bin_weights = pool.map(func, nonface_integral_imgs)
    new_weights[1] = weights[1] * np.array(np.exp(score_bin_weights))

    # go back through new weights and normalize
    norm_factor = sum(new_weights[0]) + sum(new_weights[1])
    new_weights /= norm_factor

    return new_weights, (bin_boundaries, bin_weights)


def calculate_single_bin_score(img, feature, bin_weights, bin_boundaries):

    score = feature.evaluate(img)
    bin_index = find_bin_index(bin_boundaries, score)
    return bin_weights[bin_index]


def find_bin_index(bins, x):

    if x <= bins[0]:
        return 0

    for i in range(1, len(bins)):
        if bins[i-1] < x <= bins[i]:
            return i

    if x >= bins[-1]:
        return len(bins) - 1


def run_classifier(image, classifier, classify=False):
    # for each weak classifier, score the window and find which bin it falls in to give appropriate response/bin weight
    response = 0
    for x in classifier:
        score = x.evaluate(image)
        bin_boundaries, bin_weights = x.weight
        bin_index = find_bin_index(bin_boundaries, score)  # x.weight stores bin weights
        response += bin_weights[bin_index]
    if classify:
        return np.sign(response)
    else:
        return response


def graph_classifier_performance(classifier, num_features, face_integral_imgs, nonface_integral_imgs, filename='plot1.png'):

    fig = plt.figure()

    sub_classifier = classifier[:num_features]

    face_scores = [run_classifier(img, sub_classifier, classify=False) for img in face_integral_imgs]
    nonface_scores = [run_classifier(img, sub_classifier, classify=False) for img in nonface_integral_imgs]

    sns.distplot(face_scores, hist=False, label='face')
    sns.distplot(nonface_scores, hist=False, label='non-face')
    plt.legend()
    plt.title('Strong classifier distribution at T = ' + str(num_features))
    plt.xlabel('score')
    plt.ylabel('frequency')

    fig.savefig(filename)


def roc_curve(classifier, num_features, face_integral_imgs, nonface_integral_imgs, filename='plot1.png'):

    sub_classifier = classifier[:num_features]

    face_scores = [run_classifier(img, sub_classifier, classify=False) for img in face_integral_imgs]
    nonface_scores = [run_classifier(img, sub_classifier, classify=False) for img in nonface_integral_imgs]

    # calculate final classification based on sum of all feature scores, convert each score to tuple and add 1 for
    # positive samples and -1 for negative samples
    face_final_scores = [(score, 1) for score in face_scores]
    nonface_final_scores = [(score, -1) for score in nonface_scores]

    # calculate direction of threshold
    pos_median = np.median([x[0] for x in face_final_scores])
    neg_median = np.median([x[0] for x in nonface_final_scores])
    if pos_median < neg_median:
        parity = 1
    else:
        parity = -1

    # combine and sort by score
    samples = face_final_scores + nonface_final_scores
    samples.sort()

    pos_total = float(len(face_final_scores))
    neg_total = float(len(nonface_final_scores))

    # establishing a threshold will split the sorted samples into two halves. We can try each partition, calculate how
    # many points are correctly classified, and choose the best threshold with the lowest error
    pos_below_thresh = 0
    neg_below_thresh = 0

    # for each threshold, calculate false positive rate (# false positives / total negatives) and true positive
    # rate (# true positives / total positives)
    fprs = []
    tprs = []
    for i in range(len(samples)):
        if samples[i][1] == 1:
            pos_below_thresh += 1
        else:
            neg_below_thresh += 1

        if parity == 1: # call all samples below threshold as faces
            true_positive = pos_below_thresh
            false_positive = neg_below_thresh
        else:  # call all samples above threshold as faces
            true_positive = pos_total - pos_below_thresh
            false_positive = neg_total - neg_below_thresh
        tprs.append(true_positive / pos_total)
        fprs.append(false_positive / neg_total)

    fig = plt.figure()
    plt.title('ROC curve at T = ' + str(num_features))
    # draw straight xy line
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), color='black')
    # draw roc curve
    plt.plot(fprs, tprs, color='black')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')

    fig.savefig(filename)
