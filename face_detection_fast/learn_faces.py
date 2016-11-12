# This script trains the face detection algorithm using Adaboost to train 100 feature classifier based on weak Haar
# classifiers. These 100 features are refined using 100 iterations of Realboost so each feature is refined. Then, a
# sliding window approach at multiple scales is used to detect faces in the test images. First, face detection is run
# on the background images and any identified faces are added as hard negatives to the training set. Realboost is
# trained again with these added hard negatives, using weights from the last iteration of Realboost and continuing for
# another 100 so all features are tuned. The re-trained classifier is run again on the test images and any overlapping
# boxes are reduced with non-maximum suppression.

import sys
sys.path.extend(['/Users/Kimberly/Documents/machine_learning/project2_python/face_detection_fast'])
import os
os.chdir('./face_detection_fast')

from scipy.misc import imread
from skimage.transform import integral_image
import face_haar
import adaboost
import realboost
import os
import numpy as np
import time
import pickle
import multiprocessing
import detect_faces


def calculate_integral_imgs(images):
    # calculate integral images and pad 0th column and 0th row
    pool = multiprocessing.Pool(processes=30)
    integral_imgs = np.array(pool.map(integral_image, images))
    integral_imgs = np.pad(integral_imgs, pad_width=((0, 0), (1, 0), (1, 0)), mode='constant', constant_values=0)
    return integral_imgs

if __name__ == '__main__':
    print "Reading in training images..."
    folder = '/Users/Kimberly/Documents/machine_learning/project2/'
    # folder = '/data/home/kinsigne/ml/project2_python/'
    face_folder = folder + 'newface16'
    face_images = [imread(os.path.join(face_folder, img_file), flatten=True)
                   for img_file in os.listdir(face_folder) if img_file.endswith('bmp')]

    nonface_folder = folder + 'nonface16'
    nonface_images = [imread(os.path.join(nonface_folder, img_file), flatten=True)
                      for img_file in os.listdir(nonface_folder) if img_file.endswith('bmp')]

    face_integral_imgs = calculate_integral_imgs(face_images)
    nonface_integral_imgs = calculate_integral_imgs(nonface_images)

    print "Creating initial weak Haar features..."
    start = time.time()
    weak_haar_features = face_haar.create_weak_haar(face_integral_imgs, nonface_integral_imgs, complexity='medium')
    end = time.time()
    print end - start
    # save results to pickle file
    pickle.dump(weak_haar_features, open('weak_haar_features.pkl', 'wb'))

    print "Boosting features with Adaboost..."
    start = time.time()
    boosted_features = adaboost.adaboost(weak_haar_features, 100, face_integral_imgs, nonface_integral_imgs)
    end = time.time()
    print end - start

    # graph classifier performance
    steps = [10, 50, 100]
    for step in steps:
        adaboost.graph_classifier_performance(boosted_features, step, face_integral_imgs, nonface_integral_imgs,
                                              'classifier_T' + str(step) + '.png')

    # graph ROC curves
    for step in steps:
        adaboost.roc_curve(boosted_features, step, face_integral_imgs, nonface_integral_imgs,
                           'classifier_T' + str(step) + '_roc.png')

    # graph top ten boosted features with highest weights
    sample_face = imread(os.path.join(face_folder, 'face16_000001.bmp'))
    face_haar.graph_feature_image(sorted(boosted_features, key=lambda x: x.weight, reverse=True)[:10], sample_face,
                                  num_row=2, num_col=5, filename='feature_imgs.png')

    # save results to pickle file
    pickle.dump(boosted_features, open('boosted_features.pkl', 'wb'))

    print "Boosting top 100 features from Adaboost with Realboost..."
    # run realboost on 100 features from adaboost
    start = time.time()
    real_boosted_features = realboost.realboost(boosted_features, 100, face_integral_imgs, nonface_integral_imgs)
    end = time.time()
    print end - start
    # save boosted features
    pickle.dump(real_boosted_features, open('real_boosted_features.pkl', 'wb'))

    # graph classifier performance
    for step in steps:
        realboost.graph_classifier_performance(real_boosted_features, step, face_integral_imgs, nonface_integral_imgs,
                                               'classifier_T' + str(step) + '_realboost.png')

    # graph ROC performance
    for step in steps:
        realboost.roc_curve(real_boosted_features, step, face_integral_imgs, nonface_integral_imgs,
                            'classifier_T' + str(step) + '_realboost_roc.png')

    # read in test and background images and convert to grayscale
    bg1 = imread('./Test_and_background_Images/Background_1.jpg', flatten=True)
    bg2 = imread('./Test_and_background_Images/Background_2.jpg', flatten=True)
    bg3 = imread('./Test_and_background_Images/Background_3.jpg', flatten=True)
    test_img1 = imread('./Test_and_background_Images/Test_Image_1.jpg', flatten=True)
    test_img2 = imread('./Test_and_background_Images/Test_Image_2.jpg', flatten=True)
    # keep RGB version for drawing
    bg1_orig = imread('./Test_and_background_Images/Background_1.jpg')
    bg2_orig = imread('./Test_and_background_Images/Background_2.jpg')
    bg3_orig = imread('./Test_and_background_Images/Background_3.jpg')
    test_img1_orig = imread('./Test_and_background_Images/Test_Image_1.jpg')
    test_img2_orig = imread('./Test_and_background_Images/Test_Image_2.jpg')

    # detect faces on test images with  classifier
    min_scale, max_scale, step = 0.06, 0.22, 0.01
    img1_faces = detect_faces.detect_faces(test_img1, real_boosted_features, min_scale, max_scale, step)
    img2_faces = detect_faces.detect_faces(test_img2, real_boosted_features, min_scale, max_scale, step)
    # draw boxes
    detect_faces.graph_boxes_on_image(img1_faces, test_img1_orig, 'boxed_faces_all_noneg__test_img1.png')
    detect_faces.graph_boxes_on_image((img2_faces, test_img2_orig, 'boxed_faces_all_noneg_test_img2.png'))

    # detect faces on background images to use as hard negatives and re-train Realboost classifier, starting with
    # weights from last iteration of Realboost training
    weights = pickle.load(open('realboost_weights.pkl', 'rb'))
    retrained = detect_faces.hard_negative_mining([bg1, bg2, bg3], real_boosted_features, weights,
                                                  face_integral_imgs, nonface_integral_imgs, min_scale, max_scale, step)

    # detect faces on test images with re-trained classifier
    img1_faces = detect_faces.detect_faces(test_img1, retrained, min_scale, max_scale, step)
    img2_faces = detect_faces.detect_faces(test_img2, retrained, min_scale, max_scale, step)

    # draw boxes
    detect_faces.graph_boxes_on_image(img1_faces, test_img1_orig, 'boxed_faces_all_retrain_test_img1.png')
    detect_faces.graph_boxes_on_image((img2_faces, test_img2_orig, 'boxed_faces_all_retrain_test_img2.png'))

    # reduce number of overlapping hits
    threshold = 0.50
    img1_faces_reduced = detect_faces.nonmax_suppression(img1_faces, threshold)
    img2_faces_reduced = detect_faces.nonmax_suppression(img2_faces, threshold)

    # draw boxes
    detect_faces.graph_boxes_on_image(img1_faces_reduced, test_img1_orig, 'boxed_faces_suppress_test_img1.png')
    detect_faces.graph_boxes_on_image((img2_faces_reduced, test_img2_orig, 'boxed_faces_suppress_test_img2.png'))