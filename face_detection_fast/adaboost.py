from math import log10
from functools import partial
from adaboost_funcs import calculate_weighted_error_pairs, calculate_weighted_error
from haar_funcs import determine_threshold
from itertools import chain
import multiprocessing
# import face_haar
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time


def adaboost(features, num_iterations, face_integral_imgs, nonface_integral_imgs):
    """
    :param features: list of all Haar features
    :param num_iterations: number of iterations
    :param face_integral_imgs: positive training images
    :param nonface_integral_imgs: negative training images
    :return: boosted_features
    """

    num_processes = 30

    # get rid of any classifiers with error = 50%
    # for classifiers with error > 50%, flip parity to make error < 50%
    # give each feature a weight of 1 to calculate empirical error

    print "Throwing out random classifiers, adjusting sign so error < 50% ..."

    m = len(face_integral_imgs) + len(nonface_integral_imgs)
    h = []  # weak classifiers
    emp_weights = [np.ones((1, len(face_integral_imgs)))[0], np.ones((1, len(nonface_integral_imgs)))[0]]

    pool = multiprocessing.Pool(processes=num_processes)
    # func = partial(calculate_weighted_error, face_integral_imgs=face_integral_imgs,
    #                nonface_integral_imgs=nonface_integral_imgs, weights=emp_weights)
    # errors = pool.map(func, features)

    features = [(features[i], features[i+len(features)/2]) for i in xrange(len(features)/2)]
    feature_pairs = features
    func = partial(calculate_weighted_error_pairs, face_integral_imgs=face_integral_imgs,
                   nonface_integral_imgs=nonface_integral_imgs, weights=emp_weights)
    errors = pool.map(func, feature_pairs)

    print "Done calculating errors, adjusting feature signs..."

    # Keep classifiers with error != 50%, if error is greater than 50%, flip the parity
    for i in range(len(errors)):
        feature, inverse_feature = copy.copy(feature_pairs[i])
        error, inverse_error = errors[i]
        feature_pair = [None, None]
        if round(error, 4) != 0.50:
            if error > 0.50:
                feature.parity *= -1
            feature_pair[0] = feature
        if round(inverse_error) != 0.50:
            if inverse_error > 0.50:
                inverse_feature.parity *= -1
            feature_pair[1] = inverse_feature
        if not all(x is None for x in feature_pair):
            # append as tuple
            h.append((feature_pair[0], feature_pair[1]))

    print "Number of classifiers with error != 50% :", len(h)

    # set up graph
    graph_steps = [0, 10, 50, 100]
    fig = plt.figure()
    plt.xlabel('classifier number')
    plt.ylabel('error rate')
    plt.title('Error of top 1000 weak classifiers')

    # initialize uniform weight for each data point
    initial_weight = 1 / float(m)
    weights = [[np.array([initial_weight] * len(face_integral_imgs)),
               np.array([initial_weight] * len(nonface_integral_imgs))]]

    boosted_features = []

    for t in range(num_iterations):
        print "Iteration ", str(t)
        start = time.time()

        # compute weighted error for each weak classifier
        func = partial(calculate_weighted_error_pairs, face_integral_imgs=face_integral_imgs,
                       nonface_integral_imgs=nonface_integral_imgs, weights=weights[t])
        errors = pool.map(func, h)

        # choose classifier with lowest weighted error, flatten errors to get exact index
        min_error, min_idx = min((val, idx) for (idx, val) in enumerate(chain(*errors)))
        # grab appropriate element from list of tuples
        feature_pair = list(h.pop(min_idx/2))
        h_new = feature_pair[min_idx%2]
        # set selected element to none and add back into list of features if both are not None
        feature_pair[min_idx%2] = None
        if not all(x is None for x in feature_pair):
            h.append((feature_pair[0], feature_pair[1]))

        print min_error
        print h_new.__dict__

        # assign weight for new classifier
        alpha = 0.5*log10((1-min_error)/min_error)
        h_new.weight = alpha
        boosted_features.append(h_new)

        # update all weights
        weights.append(update_weights(alpha, h_new, face_integral_imgs, nonface_integral_imgs, weights[t]))

        # update feature thresholds based on new weights
        # func = partial(update_threshold, face_integral_imgs=face_integral_imgs,
        #                nonface_integral_imgs=nonface_integral_imgs, weights=weights[t+1])
        func = partial(update_threshold_pairs, face_integral_imgs=face_integral_imgs,
                       nonface_integral_imgs=nonface_integral_imgs, weights=weights[t + 1])
        h = pool.map(func, h)

        # graph error rate for remaining top 1000 weak classifiers
        if t in graph_steps:
            errors = list(chain(*errors))
            errors.sort()
            plt.plot(errors[:1000], label='T = '+str(t))

        end = time.time()
        print "Time for last loop:", end - start

    # plot errors for last iteration
    errors = list(chain(*errors))
    plt.plot(errors[:1000], label='T = 100')
    plt.legend()
    fig.savefig('weak_classifier_error.png')

    pool.close()

    return boosted_features


def update_weights(alpha, feature, face_integral_imgs, nonface_integral_imgs, weights):
    """
    :param alpha: weight of newly added classifier
    :param feature: newly added classifier
    :param face_integral_imgs: positive integral images
    :param nonface_integral_imgs: negative integral images
    :param weights: all weights, ith entry are weights at ith iteration. Each entry is two element list where first
    entry is positive weights and second entry is negative weights
    :return: updated weights
    """

    new_weights = [[], []]
    guesses = [feature.classify(img) for img in face_integral_imgs]
    adjust = np.array([np.exp(-alpha) if x == 1 else np.exp(alpha) for x in guesses])
    new_weights[0] = weights[0] * adjust

    guesses = [feature.classify(img) for img in nonface_integral_imgs]
    adjust = np.array([np.exp(-alpha) if x == -1 else np.exp(alpha) for x in guesses])
    new_weights[1] = weights[1] * adjust

    norm_factor = sum(new_weights[0]) + sum(new_weights[1])
    new_weights /= norm_factor

    return new_weights


def update_threshold(feature, face_integral_imgs, nonface_integral_imgs, weights):

    # calculate distribution of scores
    scores = [[feature.evaluate(img) for img in face_integral_imgs],
              [feature.evaluate(img) for img in nonface_integral_imgs]]

    updated_feature = determine_threshold(feature, scores, weights)

    return updated_feature


def update_threshold_pairs(feature_pair, face_integral_imgs, nonface_integral_imgs, weights):

    feature, inverse_feature = feature_pair
    updated_feature = None
    updated_inverse_feature = None

    if feature:
        # calculate distribution of scores
        scores = [np.array([feature.evaluate(img) for img in face_integral_imgs]),
                  np.array([feature.evaluate(img) for img in nonface_integral_imgs])]
        updated_feature = determine_threshold(feature, scores, weights)
        if inverse_feature:
            scores = [scores[0] * -1, scores[1] * -1]
            updated_inverse_feature = determine_threshold(inverse_feature, scores, weights)
    else:
        updated_inverse_feature = update_threshold(feature, face_integral_imgs, nonface_integral_imgs)

    return updated_feature, updated_inverse_feature


def graph_classifier_performance(classifier, num_features, face_integral_imgs, nonface_integral_imgs, filename='plot1.png'):

    fig = plt.figure()

    sub_classifier = classifier[:num_features]

    face_step_scores = [[feature.classify(img)*feature.weight for feature in sub_classifier]
                        for img in face_integral_imgs]
    nonface_step_scores = [[feature.classify(img)*feature.weight for feature in sub_classifier]
                           for img in nonface_integral_imgs]

    # calculate final classifier score based on sum of all feature scores
    face_final_scores = [sum(x) for x in face_step_scores]
    nonface_final_scores = [sum(x) for x in nonface_step_scores]

    sns.distplot(face_final_scores, hist=False, label='face')
    sns.distplot(nonface_final_scores, hist=False, label='non-face')
    plt.legend()
    plt.title('Strong classifier distribution at T = ' + str(num_features))
    plt.xlabel('score')
    plt.ylabel('frequency')

    fig.savefig(filename)


def roc_curve(classifier, num_features, face_integral_imgs, nonface_integral_imgs, filename='plot1.png'):

    sub_classifier = classifier[:num_features]

    face_step_scores = [[feature.classify(img)*feature.weight for feature in sub_classifier]
                        for img in face_integral_imgs]
    nonface_step_scores = [[feature.classify(img)*feature.weight for feature in sub_classifier]
                           for img in nonface_integral_imgs]

    # calculate final classification based on sum of all feature scores, convert each score to tuple and add 1 for
    # positive samples and -1 for negative samples
    face_final_scores = [(sum(x), 1) for x in face_step_scores]
    nonface_final_scores = [(sum(x), -1) for x in nonface_step_scores]

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
