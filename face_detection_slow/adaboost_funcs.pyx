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


