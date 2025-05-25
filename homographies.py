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
    """
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
    Ulaz:- source_points: (n, 2) - tocke iz slike objekta- target_points: (n, 2) - tocke u slici scene- confidence: float (npr. 0.85)- inlier_threshold: float - maksimalna dozvoljena udaljenost za inlier
 Izlaz:- best_homography: (3, 3)- best_inliers: boolean array (n,)- num_iterations: broj iteracija prema formuli
 Upute:
 1. Svaka iteracija uzima slucajna 4 para tocaka.
 2. Racuna se homografija pomocu find_homography_leastsquares.
 3. Transformiraju se sve tocke iz objekta.
 4. Racunaju se udaljenosti do ciljanih tocaka.
 5. Biljeze se koje su tocke inlieri (udaljenost < prag).
 6. Homografija s najvise inliera se zapamti.
 7. Na kraju se ponovno racuna homografija samo s inlierima.
 Broj iteracija:
 num_iterations = int(log(1 - confidence) / log(1 - (4/n) * ((4 - 1)/(n - 1))))

"""
def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Ulaz:- source_points: numpy array oblika (n, 2) - tocke iz slike objekta- target_points: numpy array oblika (n, 2) - pripadne tocke u slici scene
 Izlaz:- homography: numpy array oblika (3, 3) - homografijska matrica koja preslikava source_points u target_points
 Zadatak:
 1. Mora se koristiti najmanje 4 para tocaka.
 2. Za svaki par tocka (x, y) -> (x', y') dodaj 2 jednadzbe u sustav:
   x' = (h1*x + h2*y + h3) / (h7*x + h8*y + 1)
   y' = (h4*x + h5*y + h6) / (h7*x + h8*y + 1)
 3. Prevesti jednadzbe u oblik A * h = b
 4. Rjesiti sustav koristeci np.linalg.lstsq(...)
 5. Dodati 1 kao deveti element i reshapeati rezultat u 3x3 matricu.
 """

