import cv2
import numpy as np
from pathlib import Path
import time

from homographies import find_homography_ransac
from helper_functions import show_image, filter_matches

# ----------------------------
# Parametri
feature_type = "ORB"  # ili "SIFT"
ransac_confidence = 0.85
ransac_inlier_threshold = 5.0

current_path = Path(__file__).parent
object_folder = current_path / "pojedinacne_slike"
scene_folder = current_path / "database"
mask_folder = current_path / "evaluacija" / "output_maske"

# Rezultat output fajl
output_path = current_path / f"evaluacija_matcheva_{feature_type}.txt"
output_lines = []

# ----------------------------
# Petlja po objektima 1–6
for obj_index in range(1, 7):
    object_path = object_folder / f"{obj_index}.png"
    object_img = cv2.imread(str(object_path), cv2.IMREAD_GRAYSCALE)
    if object_img is None:
        print(f"⚠️  Objekt {object_path.name} nije pronađen.")
        continue

    # Detektor i matcher
    if feature_type == "SIFT":
        detector = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        use_knn = True
    elif feature_type == "ORB":
        detector = cv2.ORB_create(nfeatures=1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        use_knn = False
    else:
        raise ValueError("Nepodržani feature_type. Koristi 'SIFT' ili 'ORB'.")

    source_kp, source_des = detector.detectAndCompute(object_img, None)

    # ----------------------------
    # Petlja po scenama
    scene_paths = sorted(scene_folder.glob("*.png"))

    for scene_path in scene_paths:
        scene_img = cv2.imread(str(scene_path), cv2.IMREAD_GRAYSCALE)
        if scene_img is None:
            continue

        target_kp, target_des = detector.detectAndCompute(scene_img, None)

        if source_des is None or target_des is None:
            continue

        if use_knn:
            matches_raw = matcher.knnMatch(source_des, target_des, k=2)
            matches = filter_matches(matches_raw)
        else:
            matches = matcher.match(source_des, target_des)
            matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            continue

        # Evaluacija maske
        scene_index = int(scene_path.stem.split('.')[0])
        mask_path_npy = mask_folder / f"scena_{scene_index}_mask.npy"
        mask_path_png = mask_folder / f"scena_{scene_index}_mask.png"

        if mask_path_npy.exists():
            mask = np.load(mask_path_npy)
        elif mask_path_png.exists():
            mask = cv2.imread(str(mask_path_png), cv2.IMREAD_GRAYSCALE)
        else:
            continue

        in_object = 0
        for match in matches:
            x, y = target_kp[match.trainIdx].pt
            x, y = int(round(x)), int(round(y))

            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] == obj_index:
                    in_object += 1

        total_matches = len(matches)
        ratio = in_object / total_matches if total_matches > 0 else 0

        # Spremi rezultat
        output_lines.append(
            f"objekt={obj_index} scena={scene_index} ukupno_matcheva={total_matches} "
            f"u_objektu={in_object} postotak={ratio * 100:.2f}%"
        )

# ----------------------------
# Spremi sve rezultate u fajl
with open(output_path, 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"[✓] Rezultati spremljeni u: {output_path}")
