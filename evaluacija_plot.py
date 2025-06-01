import matplotlib.pyplot as plt
import re
from collections import defaultdict

# 🔧 Putanje do evaluacijskih fajlova
sift_file = "evaluacija_matcheva.txt"
orb_file = "evaluacija_matcheva_ORB.txt"

# 🔍 Funkcija za parsiranje
def parse_file(filepath):
    data = defaultdict(lambda: [0, 0])  # objekt_id: [u_objektu, ukupno]
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"objekt=(\d+)\s+scena=\d+\s+ukupno_matcheva=(\d+)\s+u_objektu=(\d+)", line)
            if match:
                obj = int(match[1])
                total = int(match[2])
                inside = int(match[3])
                data[obj][0] += inside
                data[obj][1] += total
    return data

# 📊 Učitaj podatke
sift_data = parse_file(sift_file)
orb_data = parse_file(orb_file)

# 📈 Crtanje grafova
fig, axes = plt.subplots(2, 6, figsize=(18, 7))
fig.suptitle("Postotak točnih uparivanja po objektima (SIFT vs ORB)", fontsize=16)

for obj_id in range(1, 7):
    for row, (dataset, label) in enumerate(zip([sift_data, orb_data], ["SIFT", "ORB"])):
        ax = axes[row, obj_id - 1]
        if obj_id in dataset:
            inside, total = dataset[obj_id]
            outside = total - inside
            sizes = [inside, outside]
            labels = ['U objektu', 'Izvan objekta']
            colors = ['#4CAF50', '#F44336']
            explode = (0.05, 0)

            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
        ax.set_title(f"{label} - Objekt {obj_id}", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
