import sys
import os
import cv2
import torch
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Koristi backend koji ne traži GUI
import matplotlib.pyplot as plt

# Dodaje scripts folder u path da Python može pronaći module u njemu
sys.path.append(os.path.join(os.path.dirname(__file__), 'SuperGluePretrainedNetwork-master'))

from models.superpoint import SuperPoint
from models.superglue import SuperGlue


total_time_start = time.time()

# Uređaj: koristi GPU ako postoji
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Superpoint inicijalizacija
superpoint = SuperPoint({}).to(device)
superpoint.eval()
# SuperGlue inicijalizacija s indoor težinama
superglue = SuperGlue({
    'weights': 'indoor',
    'match_threshold': 0.01
}).to(device)
superglue.eval()



# Folderi i fajlovi
base_dir = os.path.dirname(os.path.dirname(__file__))
objects_dir = os.path.join(base_dir, 'pojedinacne_slike')
scenes_dir = os.path.join(base_dir, 'database')
outputs_dir = os.path.join(base_dir, 'outputs')
# Stvaranje outputs foldera ako ne postoji
os.makedirs(outputs_dir, exist_ok=True)

# Učitavanje imena fajlova (png)
object_files = sorted([f for f in os.listdir(objects_dir) if f.endswith('.png')])
scene_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith('.png')])



for obj_file in object_files:
    start_time0 = time.time()
    obj_path = os.path.join(objects_dir, obj_file)
    # Učitavanje slike
    object_img = cv2.imread(obj_path)
    # Pretvorba u grayscale
    object_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    # Tesnor slika
    object_tensor = torch.tensor(object_gray, dtype=torch.float32) / 255.0
    # Reshape
    object_tensor = object_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    end_time0 = time.time()

    # Detekcija ključnih točaka i deskriptora
    with torch.no_grad():
        object_output = superpoint({'image': object_tensor})

    for scene_file in scene_files:
        start_time1 = time.time()
        scene_path = os.path.join(scenes_dir, scene_file)
        scene_img = cv2.imread(scene_path)
        scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
        scene_tensor = torch.tensor(scene_gray, dtype=torch.float32) / 255.0
        scene_tensor = scene_tensor.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            scene_output = superpoint({'image': scene_tensor})



        # Spajanje sa SuperGlue
        with torch.no_grad():
            kpts0 = object_output['keypoints'][0]
            kpts1 = scene_output['keypoints'][0]
            desc0 = object_output['descriptors'][0]
            desc1 = scene_output['descriptors'][0]
            scores0 = object_output['scores'][0]
            scores1 = scene_output['scores'][0]

            if isinstance(kpts0, np.ndarray):
                kpts0 = torch.from_numpy(kpts0)
            if isinstance(kpts1, np.ndarray):
                kpts1 = torch.from_numpy(kpts1)

            kpts0 = kpts0.unsqueeze(0).to(device)   # [1, N, 2]
            kpts1 = kpts1.unsqueeze(0).to(device)
            desc0 = desc0.unsqueeze(0).to(device)   # [1, 256, N]
            desc1 = desc1.unsqueeze(0).to(device)
            scores0 = scores0.unsqueeze(0).to(device)   # [1, N]
            scores1 = scores1.unsqueeze(0).to(device)

            match_output = superglue({
                'keypoints0': kpts0,
                'keypoints1': kpts1,
                'descriptors0': desc0,
                'descriptors1': desc1,
                'scores0': scores0,
                'scores1': scores1,
                'image0': object_tensor,
                'image1': scene_tensor,
            })

        matches = match_output['matches0'][0].cpu().numpy()        # Indeksi parova ili -1 za nema para
        num_matches = np.sum(matches > -1)

        # Za spremanje pretvori u RGB
        object_rgb = cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB)
        scene_rgb = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
        
        # Spoji slike horizontalno
        h1, w1, _ = object_rgb.shape
        h2, w2, _ = scene_rgb.shape
        height = max(h1, h2)
        combined_img = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
        combined_img[:h1, :w1] = object_rgb
        combined_img[:h2, w1:] = scene_rgb

        # Dohvati koordinate matcheva
        kpts0_np = kpts0[0].cpu().numpy() # Ukloni batch dimenziju [N, 2]
        kpts1_np = kpts1[0].cpu().numpy()

        matched_kpts0 = kpts0_np[matches > -1]
        matched_kpts1 = kpts1_np[matches[matches > -1]]

        # Iscrtavanje 
        for pt0, pt1 in zip(matched_kpts0, matched_kpts1):
            pt0 = tuple(map(int, pt0))
            pt1 = tuple(map(int, pt1))
            pt1_shifted = (int(pt1[0] + w1), int(pt1[1])) # Pomak u desno za spajanje
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.circle(combined_img, pt0, 3, color, -1)
            cv2.circle(combined_img, pt1_shifted, 3, color, -1)
            cv2.line(combined_img, pt0, pt1_shifted, color, 1)

        # Format imena: x_y.png i x_y.txt (brojevi iz imena fajlova)
        obj_num = os.path.splitext(obj_file)[0]
        scene_num = os.path.splitext(scene_file)[0]
        output_img_path = os.path.join(outputs_dir, f"{obj_num}_{scene_num}.png")
        output_txt_path = os.path.join(outputs_dir, f"{obj_num}_{scene_num}.txt")

        plt.imsave(output_img_path, combined_img)
                   
        # Racunanje potrebnog vremena po petlji
        end_time1 = time.time()
        elapsed_time = end_time1 - start_time1 + (end_time0 - start_time0)

        # Spremi tekstualni file s info
        with open(output_txt_path, 'w') as f:
            f.write(f"Vrijeme izvodenja (sekundi): {elapsed_time:.4f}\n")
            f.write(f"Broj tocaka objekta: {len(object_output['keypoints'][0])}\n")
            f.write(f"Broj tocaka scene: {len(scene_output['keypoints'][0])}\n")
            f.write(f"Broj pronadenih podudaranja: {num_matches}\n")

# Ukupno vrijeme za 120 kombinacija
total_time_end = time.time()
print("Ukupno vrijeme proteklo: ", total_time_end - total_time_start)