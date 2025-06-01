import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import colormaps

# ---------- KONFIGURACIJA ----------
scene_cm_width = 42.0
scene_cm_height = 59.4
scene_px_width = 1587
scene_px_height = 2245
px_per_cm = scene_px_width / scene_cm_width  # ≈ 37.79

csv_path = "tocke_sve.csv"
object_img_dir = "objekti/"
output_dir = "output_maske"

os.makedirs(output_dir, exist_ok=True)

# ---------- UČITAJ CSV ----------
df = pd.read_csv(csv_path)

# ---------- UČITAJ OBJEKTE ----------
object_images = {}
for i in df["objekt"].dropna().unique():
    path = os.path.join(object_img_dir, f"{int(i)}.png")
    if os.path.exists(path):
        img = Image.open(path).convert("RGBA")
        object_images[int(i)] = img

# ---------- GENERIRAJ MASKE ----------
for scene_id in df["scena"].unique():
    scene_mask = np.zeros((scene_px_height, scene_px_width), dtype=np.uint8)

    scene_df = df[df["scena"] == scene_id].copy()
    scene_df["layer"] = scene_df["layer"].astype(int)
    scene_df = scene_df.sort_values(by="layer", ascending=True)

    for _, row in scene_df.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]) or pd.isna(row["w"]) or pd.isna(row["h"]):
            continue

        obj_id = int(row["objekt"])
        x_px = row["x"] * px_per_cm
        y_px = row["y"] * px_per_cm
        w_px = row["w"] * px_per_cm
        h_px = row["h"] * px_per_cm
        kut = float(row.get("kut", 0))

        if obj_id not in object_images:
            continue

        # Binarna maska iz alpha kanala
        alpha = np.array(object_images[obj_id].split()[-1])
        binary = (alpha > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary).resize((int(w_px), int(h_px)), resample=Image.NEAREST)

        rotated_mask = mask_img.rotate(-kut, expand=False)

        center_x = x_px + w_px / 2
        center_y = y_px + h_px / 2
        paste_x = int(center_x - rotated_mask.size[0] / 2)
        paste_y = int(center_y - rotated_mask.size[1] / 2)

        # Pretvori rotiranu masku u binarnu
        rot_arr = np.array(rotated_mask)
        mask_bin = (rot_arr > 0).astype(np.uint8)

        # Dimenzije maske
        rot_h, rot_w = mask_bin.shape

        # Koordinatni okviri u sceni
        x_start = max(0, paste_x)
        y_start = max(0, paste_y)
        x_end = min(scene_px_width, paste_x + rot_w)
        y_end = min(scene_px_height, paste_y + rot_h)

        # Odgovarajući dijelovi maske
        crop_x0 = max(0, -paste_x)
        crop_y0 = max(0, -paste_y)
        crop_x1 = crop_x0 + (x_end - x_start)
        crop_y1 = crop_y0 + (y_end - y_start)

        # Spriječi grešku ako je maska potpuno izvan scene
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            continue

        # Zalijepi u scenu
        cropped = mask_bin[crop_y0:crop_y1, crop_x0:crop_x1]
        region = scene_mask[y_start:y_end, x_start:x_end]
        region[cropped == 1] = obj_id

    # Spremi masku i vizualizaciju
    Image.fromarray(scene_mask, mode='L').save(os.path.join(output_dir, f"scena_{scene_id}_mask.png"))
    np.save(os.path.join(output_dir, f"scena_{scene_id}_mask.npy"), scene_mask)

    colored = colormaps.get_cmap("tab10")(scene_mask.astype(float) / (scene_mask.max() + 1e-6))[:, :, :3]
    colored_img = Image.fromarray((colored * 255).astype(np.uint8))
    colored_img.save(os.path.join(output_dir, f"scena_{scene_id}_mask_viz.png"))

    print(f"[✓] scena {scene_id} gotova.")
