import numpy as np


def calculate_weighted_error(feature, face_integral_imgs, nonface_integral_imgs, weights):
    """
    Calculate weighted error. If feature classification is wrong, weight by data point-specific weight. weights is a two
    element list where first entry is positive weights and second entry is negative weights.

    error(h) = weights(xi) * 1(h(xi) != yi) (for all images)

    return: feature error
    """

    cdef double error
    error = 0
    guesses = np.array([feature.classify(img) for img in face_integral_imgs])
    error += sum((guesses != 1).astype(int) * weights[0])

    guesses = np.array([feature.classify(img) for img in nonface_integral_imgs])
    error += sum((guesses != -1).astype(int) * weights[1])

    return error


def calculate_weighted_error_pairs(feature_pair, face_integral_imgs, nonface_integral_imgs, weights):
    """
    Calculate weighted error. If feature classification is wrong, weight by data point-specific weight. weights is a two
    element list where first entry is positive weights and second entry is negative weights.

    error(h) = weights(xi) * 1(h(xi) != yi) (for all images)

    return: feature error
    """

    cdef double error, inverse_error
    error = 0
    inverse_error = 0

    feature, inverse_feature = feature_pair

    if feature and inverse_feature:
        # score feature, inverse feature are same scores with flipped sign
        scores = np.array([feature.evaluate(img) for img in face_integral_imgs])
        guesses = np.array([feature.classify(face_integral_imgs[i], scores[i]) for i in xrange(len(face_integral_imgs))])
        error += sum((guesses != 1).astype(int) * weights[0])

        inverse_scores = scores * -1
        guesses = np.array([inverse_feature.classify(face_integral_imgs[i], inverse_scores[i]) for i in xrange(len(face_integral_imgs))])
        inverse_error += sum((guesses != 1).astype(int) * weights[0])

        # negative scores
        scores = np.array([feature.evaluate(img) for img in nonface_integral_imgs])
        guesses = np.array([feature.classify(nonface_integral_imgs[i], scores[i]) for i in xrange(len(nonface_integral_imgs))])
        error += sum((guesses != -1).astype(int) * weights[1])

        inverse_scores = scores * -1
        guesses = np.array( [inverse_feature.classify(nonface_integral_imgs[i], inverse_scores[i])
                             for i in xrange(len(nonface_integral_imgs))])
        inverse_error += sum((guesses != -1).astype(int) * weights[1])
        return error, inverse_error

    # if one member of pair is missing (already picked for strong classifier), use method for single feature
    if not inverse_feature:
        return None, calculate_weighted_error(inverse_feature, face_integral_imgs, nonface_integral_imgs, weights)
    if not feature:
        return calculate_weighted_error(feature, face_integral_imgs, nonface_integral_imgs, weights), None
