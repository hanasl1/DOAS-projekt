import numpy as np
import matplotlib.pyplot as plt

# Funkcija za parsiranje txt fajla
def parse_evaluation_file(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            obj = int(parts[0].split('=')[1])
            scena = int(parts[1].split('=')[1])
            postotak = float(parts[-1].split('=')[1].replace('%', ''))
            data[(obj, scena)] = postotak
    return data

# Učitaj podatke
sift_data = parse_evaluation_file("evaluacija_matcheva.txt")
orb_data = parse_evaluation_file("evaluacija_matcheva_ORB.txt")

# Pripremi matricu razlika: redovi=objekti, stupci=scene
diff_matrix = np.zeros((6, 20))  # objekti 1–6, scene 1–20

for obj in range(1, 7):
    for scena in range(1, 21):
        sift_val = sift_data.get((obj, scena), 0.0)
        orb_val = orb_data.get((obj, scena), 0.0)
        diff_matrix[obj - 1, scena - 1] = sift_val - orb_val  # razlika u postotku

# Prikaz heatmape
fig, ax = plt.subplots(figsize=(16, 6))
cax = ax.imshow(diff_matrix, cmap='bwr', aspect='auto', vmin=-100, vmax=100)

# Oznake osi
ax.set_xticks(np.arange(20))
ax.set_xticklabels([str(i) for i in range(1, 21)])
ax.set_yticks(np.arange(6))
ax.set_yticklabels([str(i) for i in range(1, 7)])
ax.set_xlabel("Scena")
ax.set_ylabel("Objekt")
ax.set_title("Razlika točnih match-eva (SIFT - ORB) u postotku")

# Dodaj vrijednosti u ćelije
for i in range(6):
    for j in range(20):
        ax.text(j, i, f"{diff_matrix[i, j]:.1f}", ha='center', va='center', color='black', fontsize=8)

# Dodaj colorbar
fig.colorbar(cax, ax=ax, label="Razlika postotaka (%)")
plt.tight_layout()
plt.show()
