#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
from helper_functions import *


def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:


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
