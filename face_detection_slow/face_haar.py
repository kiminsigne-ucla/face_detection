from itertools import chain
from Haar import *
from functools import partial
import haar_funcs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import multiprocessing


def create_weak_haar(face_integral_imgs, nonface_integral_imgs, complexity='simple'):

    if complexity == 'medium':
        features = init_medium()
    elif complexity == 'complex':
        features = init_complex()
    else:
        features = init_simple()

    # flatten
    features = list(chain(*features))

    print "Number of Haar features: ", len(features)

    # calculate positive and negative scores for each feature

    # no need to weight scores yet, so just use weights of 1 for each data point
    weights = [np.ones((1, len(face_integral_imgs)))[0], np.ones((1, len(nonface_integral_imgs)))[0]]
    
    func = partial(feature_eval_wrapper, face_integral_imgs=face_integral_imgs,
                   nonface_integral_imgs=nonface_integral_imgs, weights=weights)

    pool = multiprocessing.Pool(processes=30)
    weak_features = pool.map(func, features)

    return weak_features


def init_simple():
    rect_sizes = range(1, 2)

    features = []
    # create classes 1-4
    for x in range(1, 5):
        for i in rect_sizes:
            for j in rect_sizes:
                features.append(haar_feature_init(height=i, width=j, class_type=x, inverse=0,
                                                  target_height=16, target_width=16) +
                                haar_feature_init(height=i, width=j, class_type=x, inverse=1,
                                                  target_height=16, target_width=16))
    return features


def init_medium():

    features = []
    # create classes 1, 2, 4
    for x in [1, 2, 4]:
        rect_sizes = range(1, 8)
        for i in rect_sizes:
            for j in rect_sizes:
                features.append(haar_feature_init(height=i, width=j, class_type=x, inverse=0,
                                                  target_height=16, target_width=16) +
                                haar_feature_init(height=i, width=j, class_type=x, inverse=1,
                                                  target_height=16, target_width=16))
    # create class 3
    rect_sizes = range(1, 6)
    for i in rect_sizes:
        for j in rect_sizes:
            features.append(haar_feature_init(height=i, width=j, class_type=3, inverse=0,
                                              target_height=16, target_width=16) +
                            haar_feature_init(height=i, width=j, class_type=3, inverse=1,
                                              target_height=16, target_width=16))
    return features


def init_complex():

    features = []

    # class 1 and 2
    for i in range(1, 17):
        for j in range(1, 9):
            features.append(haar_feature_init(height=i, width=j, class_type=1, inverse=0,
                                              target_height=16, target_width=16) +
                            haar_feature_init(height=i, width=j, class_type=1, inverse=1,
                                              target_height=16, target_width=16))
            features.append(haar_feature_init(height=j, width=i, class_type=2, inverse=0,
                                              target_height=16, target_width=16) +
                            haar_feature_init(height=j, width=i, class_type=2, inverse=1,
                                              target_height=16, target_width=16))

    # class 3
    for i in range(1, 17):
        for j in range(1, 6):
            features.append(haar_feature_init(height=i, width=j, class_type=3, inverse=0,
                                              target_height=16, target_width=16) +
                            haar_feature_init(height=i, width=j, class_type=3, inverse=1,
                                              target_height=16, target_width=16))
    # class 4
    for i in range(1, 9):
        for j in range(1, 9):
            features.append(haar_feature_init(height=i, width=j, class_type=4, inverse=0,
                                              target_height=16, target_width=16) +
                            haar_feature_init(height=i, width=j, class_type=4, inverse=1,
                                              target_height=16, target_width=16))

    return features


def haar_feature_init(height, width, class_type, inverse, target_height, target_width):

    if class_type not in [1, 2, 3, 4]:
        raise ValueError('Please specify feature type as a number from 1-4')
    if inverse != 0 and inverse != 1:
        raise ValueError('Please specify inverse as 0 or 1')

    features = []

    if class_type == 1:
        for i in range(target_height - height + 1):
            for j in range(target_width - 2*width + 1):
                haar_feature = HaarClassI(height, width, class_type, inverse, i, j)
                features.append(haar_feature)

    elif class_type == 2:
        for i in range(target_height - 2*height + 1):
            for j in range(target_width - width + 1):
                haar_feature = HaarClassII(height, width, class_type, inverse, i, j)
                features.append(haar_feature)

    elif class_type == 3:
        for i in range(target_height - height + 1):
            for j in range(target_width - 3*width + 1):
                haar_feature = HaarClassIII(height, width, class_type, inverse, i, j)
                features.append(haar_feature)

    else:  # class_type == 4
        for i in range(target_height - 2*height + 1):
            for j in range(target_width - 2*width + 1):
                haar_feature = HaarClassIV(height, width, class_type, inverse, i, j)
                features.append(haar_feature)

    return features


def feature_eval_wrapper(feature, face_integral_imgs, nonface_integral_imgs, weights):
    # calculate positive and negative scores for each feature
    scores = [[feature.evaluate(img) for img in face_integral_imgs],
              [feature.evaluate(img) for img in nonface_integral_imgs]]

    updated_feature = haar_funcs.determine_threshold(feature, scores, weights)

    return updated_feature


def graph_feature_dist(feature, face_integral_imgs, nonface_integral_imgs, plotname='plot1.png'):

    fig = plt.figure()
    face_scores = [feature.evaluate(img) for img in face_integral_imgs]
    nonface_scores = [feature.evaluate(img) for img in nonface_integral_imgs]
    sns.distplot(np.array(face_scores), hist=False, label='face')
    sns.distplot(np.array(nonface_scores), hist=False, label='non-face')
    plt.xlabel('feature score')
    plt.ylabel('frequency')
    plt.legend()
    fig.savefig(plotname)


def graph_feature_image(features, image, num_row, num_col, filename='feature_imgs.png'):

    # x and y matrix coordinates are inverse relative to graph coordinates

    fig = plt.figure()

    for i in range(len(features)):

        ax = fig.add_subplot(num_row, num_col, i+1)
        plt.axis('off')
        plt.imshow(image)

        feature = features[i]

        if feature.class_type == 1:  # draw two vertical rectangles side by side, Rectangle plots at lower left corner
            if feature.inverse == 0:  # white on left, dark on right
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
            else:  # dark on left, white on right
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
        # two horizontal rectangles on top of each other
        elif feature.class_type == 2:
            if feature.inverse == 0:  # dark on top, white on bottom
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y, feature.x + feature.height), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
            else:  # white on top, dark on bottom
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y, feature.x + feature.height), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
        # three vertical rectangles side by side
        elif feature.class_type == 3:
            if feature.inverse == 0:  # white-dark-white
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + 2 * feature.width, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
            else:  # dark-white-dark
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + 2 * feature.width, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
        # 2x2 grid
        else:  # feature.class_type == 4
            if feature.inverse == 0:  # white-dark-dark-white
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x),
                                         width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y, feature.x + feature.height), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x + feature.height), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
            else:  # dark-white-white-dark
                rect = patches.Rectangle((feature.y, feature.x), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x),
                                         width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y, feature.x + feature.height), width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                rect = patches.Rectangle((feature.y + feature.width, feature.x + feature.height),
                                         width=feature.width,
                                         height=feature.height, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(rect)

    fig.savefig(filename)
