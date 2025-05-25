#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object recognition with selectable feature detector (SIFT, ORB, BRISK)
"""

from pathlib import Path
import numpy as np
import cv2
import time


from homographies import find_homography_leastsquares, find_homography_ransac
from helper_functions import *

if __name__ == '__main__':
    # Parameters
    image_nr = 1
    save_image = False
    use_matplotlib = False
    debug = True

    ransac_confidence = 0.85
    ransac_inlier_threshold = 5.

    # 🔧 Odaberi metodu: "SIFT", "ORB" ili "BRISK"
    feature_type = "SIFT"

    current_path = Path(__file__).parent

    scene_img = cv2.imread(str(current_path.joinpath("data/image")) + str(image_nr) + ".jpg")
    if scene_img is None:
        raise FileNotFoundError("Couldn't load scene image.")
    scene_img_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    object_img = cv2.imread(str(current_path.joinpath("data/object.jpg")))
    if object_img is None:
        raise FileNotFoundError("Couldn't load object image.")
    object_img_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

    # 🔍 Odabir značajki
    if feature_type == "SIFT":
        detector = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        use_knn = True
    elif feature_type == "ORB":
        detector = cv2.ORB_create(nfeatures=1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        use_knn = False
    elif feature_type == "BRISK":
        detector = cv2.BRISK_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        use_knn = False
    else:
        raise ValueError(f"Nepoznata metoda: {feature_type}")
    # ⏱️ Dodaj mjerenje vremena oko detekcije i podudaranja
    start_time = time.time()
    target_keypoints, target_descriptors = detector.detectAndCompute(scene_img_gray, None)
    source_keypoints, source_descriptors = detector.detectAndCompute(object_img_gray, None)

    # 🧩 Matchanje
    if use_knn:
        matches_raw = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
        matches = filter_matches(matches_raw)
    else:
        matches = matcher.match(source_descriptors, target_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
    # kraj detekcije i matchanja
    end_time = time.time()
    runtime = end_time - start_time

    matches_img = cv2.drawMatches(object_img, source_keypoints, scene_img, target_keypoints, matches, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_image(matches_img, f"{feature_type} Matches", save_image=save_image, use_matplotlib=use_matplotlib)


    source_points = np.array([source_keypoints[m.queryIdx].pt for m in matches])
    target_points = np.array([target_keypoints[m.trainIdx].pt for m in matches])

    homography, best_inliers, num_iterations = find_homography_ransac(source_points,
                                                                       target_points,
                                                                       confidence=ransac_confidence,
                                                                       inlier_threshold=ransac_inlier_threshold)
    #evaluacija
    num_keypoints_obj = len(source_keypoints)
    num_keypoints_scene = len(target_keypoints)
    num_matches = len(matches)
    num_inliers = int(np.sum(best_inliers))
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0

    print("\n--- Evaluacija ---")
    print(f"Metoda: {feature_type}")
    print(f"Broj ključnih točaka (objekt): {num_keypoints_obj}")
    print(f"Broj ključnih točaka (scena): {num_keypoints_scene}")
    print(f"Broj podudaranja (matches): {num_matches}")
    print(f"Broj inliera (RANSAC): {num_inliers}")
    print(f"Inlier ratio: {inlier_ratio:.2f}")
    print(f"Broj RANSAC iteracija: {num_iterations}")
    print(f"Vrijeme izvođenja (detekcija + dopasivanje): {runtime:.3f} s")

    #evaluacija
    draw_params = dict(matchesMask=best_inliers.astype(int),
                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    inlier_image = cv2.drawMatches(object_img, source_keypoints, scene_img, target_keypoints,
                                   matches, None, **draw_params)
    show_image(inlier_image, f"{feature_type} Inliers", save_image=save_image, use_matplotlib=use_matplotlib)

    #plot_img = draw_rectangles(scene_img, object_img, homography)
    show_image(scene_img, f"{feature_type} Final Result", save_image=save_image, use_matplotlib=use_matplotlib)

    transformed_object_img = cv2.warpPerspective(object_img, homography, dsize=scene_img.shape[1::-1])
    scene_img_blend = scene_img.copy()
    scene_img_blend[transformed_object_img != 0] = transformed_object_img[transformed_object_img != 0]
    show_image(scene_img_blend, f"{feature_type} Overlay Object", save_image=save_image, use_matplotlib=use_matplotlib)
