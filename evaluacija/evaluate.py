import os
import numpy as np
import pandas as pd
from pathlib import Path

# Putanje
outputs_dir = Path("outputs")
masks_dir = Path("output_maske")
output_csv = "evaluacija_rezultati.csv"

# Lista za rezultate
results = []

# Prolazak kroz sve .csv datoteke u outputs/
for file in sorted(outputs_dir.glob("*.csv")):
    name = file.stem  # npr. "1_12"
    try:
        objekt_id, scena_id = map(int, name.split("_"))
    except ValueError:
        continue  # preskoči ako nije u formatu broj_broj.csv

    df = pd.read_csv(file)
    if df.empty or not {"x", "y"}.issubset(df.columns):
        continue

    # Učitaj masku scene
    mask_path = masks_dir / f"scena_{scena_id}_mask.npy"
    if not mask_path.exists():
        continue
    mask = np.load(mask_path)

    # Pretvori koordinate u int i ograniči unutar granica slike
    x = df["x"].round().astype(int).clip(0, mask.shape[1] - 1)
    y = df["y"].round().astype(int).clip(0, mask.shape[0] - 1)

    matched_ids = mask[y, x]
    true_positives = np.sum(matched_ids == objekt_id)
    total = len(df)

    # Učitaj runtime iz txt ako postoji
    runtime = None
    txt_file = outputs_dir / f"{objekt_id}_{scena_id}.txt"
    if txt_file.exists():
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                if "vrijeme" in line.lower():
                    try:
                        runtime = float(line.strip().split()[-1])
                    except:
                        pass

    results.append({
        "objekt": objekt_id,
        "scena": scena_id,
        "total_matches": total,
        "true_positives": true_positives,
        "precision": true_positives / total if total > 0 else 0.0,
        "runtime": runtime
    })

# Spremi u CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"[✓] Rezultati spremljeni u: {output_csv}")
