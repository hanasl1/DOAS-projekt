#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find the projective transformation matrix (homography) between from a source image to a target image.

Author: Hana Šlipogor
MatrNr: 12404682
"""

import numpy as np
from helper_functions import *


def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transforamtion matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Is True if the point at the index is an inlier. Boolean array with shape (n,)
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """
    ######################################################
    # Write your own code here
    #some inspiration from: https://github.com/dastratakos/Homography-Estimation/blob/main/imageAnalysis.py and https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py

    best_suggested_homography = np.eye(3)
    best_inliers = np.full(shape=len(target_points), fill_value=True, dtype=bool)
    max_inliers = 0

    #formula for k trials from lecture materials
    num_iterations = int(np.log(1 - confidence) / np.log(1 - (4 / len(source_points)) * ((4 - 1) / (len(source_points) - 1))))

    for i in range (num_iterations):
        #find random indices of given arrays to apply RANSAC
        random_indices = np.random.choice(len(source_points), 4, replace=False)
        source_sample = source_points[random_indices]
        target_sample = target_points[random_indices]

        test_homography = find_homography_leastsquares(source_sample, target_sample)
        #transform source_points to homogeneous coordinates and apply the test homography
        transformed_source_points = np.hstack([source_points, np.ones((len(source_points), 1))]) @ test_homography.T
        #transform back from homogeneous coordinates
        transformed_source_points = transformed_source_points[:, :2] / transformed_source_points[:, 2][:, np.newaxis]

        #caluclulate distance between transformed points and actual target points
        distances = np.linalg.norm(transformed_source_points - target_points, axis=1)
        #boolean array to store values that are within the threshold
        # for inlier[i] = True there is a source_point[i] that is after transformation very similar to target_point[i]
        inliers = distances < inlier_threshold
        #sumation of all true values gives number of inliers
        num_inliers = np.sum(inliers)

        #if number of inliers is highest so far, store the inliers for further processing
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
    #find homography between points with most inliers
    best_suggested_homography = find_homography_leastsquares(source_points[best_inliers], target_points[best_inliers])

    ######################################################
    return best_suggested_homography, best_inliers, num_iterations


def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    ######################################################
    # Write your own code here
    homography = np.eye(3)
    m1 = np.zeros((2 * len(source_points), 8), dtype=np.float32)
    m2 = np.zeros((2 * len(source_points), 1), dtype=np.float32)
    for i, ((x_s, y_s), (x_t, y_t)) in enumerate(zip(source_points, target_points)):
        #for each pair (xi, yi) there are 2 rows in the matrix M1 and M2, therefore 2*i indexing is needed
        m1[2 * i] = [x_s, y_s, 1,0,0, 0,-x_t * x_s, -x_t * y_s]
        m1[2*i + 1] = [0, 0, 0, x_s, y_s, 1, -y_t*x_s, -y_t*y_s]

        m2[2* i] = x_t
        m2[2*i + 1] = y_t
    #store result vector in h, others ignore
    h, _, _, _ = np.linalg.lstsq(m1, m2, rcond=None)
    #add h22 = 1 to ensure 9  elements for 3x3 matrix shape
    homography = np.append(h, 1).reshape(3, 3)

    ######################################################
    return homography
