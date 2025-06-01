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
    feature_type = "ORB"

    current_path = Path(__file__).parent

    scene_path = current_path / "database" / "17.png"
    object_path = current_path / "pojedinacne_slike" / "4.png"

    scene_img = cv2.imread(str(scene_path))
    if scene_img is None:
        raise FileNotFoundError(f"Couldn't load scene image from {scene_path}")
    scene_img_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    object_img = cv2.imread(str(object_path))
    if object_img is None:
        raise FileNotFoundError(f"Couldn't load object image from {object_path}")
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
    #show_image(matches_img, f"{feature_type} Matches", save_image=save_image, use_matplotlib=use_matplotlib)


    source_points = np.array([source_keypoints[m.queryIdx].pt for m in matches])
    target_points = np.array([target_keypoints[m.trainIdx].pt for m in matches])

    homography, best_inliers, num_iterations = find_homography_ransac(source_points,
                                                                       target_points,
                                                                       confidence=ransac_confidence,
                                                                       inlier_threshold=ransac_inlier_threshold)
    #############333
    h_obj, w_obj = object_img.shape[:2]

    corners_object = np.array([
        [0, 0],  # gornji lijevi
        [w_obj - 1, 0],  # gornji desni
        [w_obj - 1, h_obj - 1],  # donji desni
        [0, h_obj - 1]  # donji lijevi
    ], dtype=np.float32)

    corners_object_hom = cv2.perspectiveTransform(corners_object.reshape(1, -1, 2), homography)
    corners_scene = corners_object_hom[0]  # [4, 2] oblik

    # Ispis koordinata u slici scene
    print("\n--- Koordinate transformiranog objekta u slici scene ---")
    for i, (x, y) in enumerate(corners_scene):
        print(f"Kut {i + 1}: ({x:.2f}, {y:.2f})")

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
    #show_image(scene_img, f"{feature_type} Final Result", save_image=save_image, use_matplotlib=use_matplotlib)

    transformed_object_img = cv2.warpPerspective(object_img, homography, dsize=scene_img.shape[1::-1])
    scene_img_blend = scene_img.copy()
    scene_img_blend[transformed_object_img != 0] = transformed_object_img[transformed_object_img != 0]
    #show_image(scene_img_blend, f"{feature_type} Overlay Object", save_image=save_image, use_matplotlib=use_matplotlib)
    ########################################
    # ----------------------------
    # 🔍 Učitavanje maske i evaluacija poklapanja s objektom
    from pathlib import Path

    # Pronađi indeks iz imena scene
    scene_index = int(scene_path.stem.split('.')[0])  # ako se zove "3.png" -> 3
    mask_dir = current_path / "evaluacija" / "output_maske"
    mask_npy = mask_dir / f"scena_{scene_index}_mask.npy"
    mask_png = mask_dir / f"scena_{scene_index}_mask.png"

    # Učitaj masku: prioritet .npy, inače .png
    if mask_npy.exists():
        mask = np.load(mask_npy)
    elif mask_png.exists():
        mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
    else:
        print(f"[!] Maska za scenu {scene_index} nije pronađena.")
        mask = None

    if mask is not None:
        # Broji koliko match-eva (točaka iz scene) upada u masku (gdje je mask == 0)
        in_mask = 0
        for match in matches:
            x, y = target_keypoints[match.trainIdx].pt
            x, y = int(round(x)), int(round(y))

            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] == 3:  # unutra u objektu
                    in_mask += 1

        total_matches = len(matches)
        ratio = in_mask / total_matches if total_matches > 0 else 0.0

        print("\n--- Evaluacija maske ---")
        print(f"Match-eva unutar maske objekta (mask == 0): {in_mask} / {total_matches}")
        print(f"Postotak match-eva unutar objekta: {ratio * 100:.2f}%")
