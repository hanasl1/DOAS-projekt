import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
gc.collect()
import torch
import torchvision.transforms as T
from torchvision.ops import nms
from PIL import Image
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import time
import gc

# CONFIG
print('Configuring...')

### dataset
SINGLES_DIR = input('Directory with single images (default: ./data/single): ')
COMPOSITES_DIR = input('Directory with composite images / scenes (default: ./data/database): ')
if not SINGLES_DIR: SINGLES_DIR = './data/single'
if not COMPOSITES_DIR: COMPOSITES_DIR = './data/database'

### device
DEVICE = torch.device('cpu') # cuda or cpu
print(f"\tusing device: {DEVICE}")

### DINOv2
DINOV2_MODEL_NAME = 'facebook/dinov2-base'
print(f"\tloading DINOv2 model: {DINOV2_MODEL_NAME} for {DEVICE}...")
try:
    dino_image_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME, use_fast=True)
except Exception:
    print("\t\tFailed to load fast DINOv2 processor, falling back.")
    dino_image_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME, use_fast=False)
dino_model = AutoModel.from_pretrained(DINOV2_MODEL_NAME).to(DEVICE).eval()
print("\t\tDINOv2 model loaded")

### SAM
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_PATH = f"sam_{SAM_MODEL_TYPE}.pth"
if SAM_CHECKPOINT_PATH and os.path.exists(SAM_CHECKPOINT_PATH):
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE).eval()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100
    )
    sam_loaded = True
    print("\t\tSAM model loaded")
else:
    sam_loaded = False
    mask_generator = None
    print(f"\tSAM checkpoint not found. Patch matching will be used.")

### results
BASE_RESULTS_DIR = './results'
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)


# Helpers
def preprocess_for_dino(image_input, processor):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
        images_to_process = [image]
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)).convert('RGB')
        images_to_process = [image]
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
        images_to_process = [image]
    elif isinstance(image_input, list) and all(isinstance(img, Image.Image) for img in image_input):
        images_to_process = [img.convert('RGB') for img in image_input]
    else:
        raise ValueError("Unsupported image_input type for preprocess_for_dino.")
    inputs = processor(images=images_to_process, return_tensors="pt")
    return inputs['pixel_values'].to(DEVICE), images_to_process[0] if len(images_to_process) == 1 else images_to_process

@torch.no_grad()
def get_dino_embedding(pixel_values, model):
    outputs = model(pixel_values)
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        embedding = outputs.pooler_output
    elif hasattr(outputs, 'last_hidden_state'):
        embedding = outputs.last_hidden_state[:, 0]
    else:
        raise ValueError("Could not extract embedding from DINOv2 model output.")
    return embedding.cpu().numpy()

def generate_scaled_patches(image_pil, query_w, query_h,
                            scales=[0.75, 1.0, 1.25, 1.5],
                            min_patch_dim=32, scene_max_patch_ratio=0.9):
    patches_info = []
    scene_w, scene_h = image_pil.size
    for scale in scales:
        patch_w_ideal = int(query_w * scale)
        patch_h_ideal = int(query_h * scale)
        patch_w = max(min_patch_dim, min(patch_w_ideal, int(scene_w * scene_max_patch_ratio)))
        patch_h = max(min_patch_dim, min(patch_h_ideal, int(scene_h * scene_max_patch_ratio)))
        if patch_w == 0 or patch_h == 0 or patch_w > scene_w or patch_h > scene_h:
            continue
        stride_x = max(1, patch_w // 4)
        stride_y = max(1, patch_h // 4)
        for y in range(0, scene_h - patch_h + 1, stride_y):
            for x in range(0, scene_w - patch_w + 1, stride_x):
                box = (x, y, x + patch_w, y + patch_h)
                patch = image_pil.crop(box)
                patches_info.append({'patch_pil': patch, 'coords': (x, y, patch_w, patch_h)})
    return patches_info

def draw_match_info_on_image(image_cv, match_info, color=(0, 255, 0), thickness=2, text_on=True):
    if match_info is None:
        return image_cv
    img_to_draw_on = image_cv.copy()
    x, y, w, h = map(int, match_info['coords'])
    cv2.rectangle(img_to_draw_on, (x, y), (x + w, y + h), color, thickness)
    if text_on and 'score' in match_info:
        label = f"S: {match_info['score']:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness // 2 + 1)
        cv2.rectangle(img_to_draw_on, (x, y - label_height - baseline - 2), (x + label_width, y - baseline + 2), color, -1)
        cv2.putText(img_to_draw_on, label, (x, y - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if sum(color)>300 else (255,255,255) , thickness // 2 + 1)
    return img_to_draw_on

def apply_nms_to_matches(matches_info, iou_threshold=0.3):
    if not matches_info: return []
    boxes_xyxy = torch.tensor([[m['coords'][0], m['coords'][1],
                                m['coords'][0] + m['coords'][2], m['coords'][1] + m['coords'][3]]
                               for m in matches_info], dtype=torch.float32)
    scores = torch.tensor([m['score'] for m in matches_info], dtype=torch.float32)
    keep_indices = nms(boxes_xyxy, scores, iou_threshold)
    return [matches_info[i] for i in keep_indices.cpu().numpy()]

# Approach 1: Patches (slow and less accurate)
def match_with_dinov2_patches(query_embedding_data, scene_image_path,
                              similarity_threshold=0.80,
                              nms_iou_threshold=0.2,
                              patch_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                              batch_size=16):
    
    query_embedding, q_w, q_h = query_embedding_data

    time_patch_gen_start = time.time()
    _, scene_pil = preprocess_for_dino(scene_image_path, dino_image_processor)
    all_patches_info = generate_scaled_patches(scene_pil, q_w, q_h, scales=patch_scales)
    time_patch_gen_end = time.time()
    patch_gen_time = time_patch_gen_end - time_patch_gen_start

    original_scene_cv_bgr = cv2.cvtColor(np.array(scene_pil), cv2.COLOR_RGB2BGR)

    if not all_patches_info:
        return None, -1.0, original_scene_cv_bgr, patch_gen_time, 0.0

    candidate_matches = []
    processed_count = 0
    max_score_overall = -1.0
    
    time_matching_start = time.time()
    for i in range(0, len(all_patches_info), batch_size):
        batch_patch_info = all_patches_info[i:i+batch_size]
        batch_patch_pils = [info['patch_pil'] for info in batch_patch_info]
        patch_batch_pixel_values, _ = preprocess_for_dino(batch_patch_pils, dino_image_processor)
        patch_embeddings = get_dino_embedding(patch_batch_pixel_values, dino_model)
        similarities = cosine_similarity(query_embedding, patch_embeddings)[0]
        
        for j, score in enumerate(similarities):
            max_score_overall = max(max_score_overall, float(score))
            if score >= similarity_threshold:
                candidate_matches.append({'coords': batch_patch_info[j]['coords'], 'score': float(score)})
        processed_count += len(batch_patch_pils)

    final_matches_after_nms = apply_nms_to_matches(candidate_matches, nms_iou_threshold)
    final_matches_after_nms.sort(key=lambda x: x['score'], reverse=True)
    best_match_info = final_matches_after_nms[0] if final_matches_after_nms else None
    time_matching_end = time.time()
    matching_time = time_matching_end - time_matching_start

    if best_match_info:
        return best_match_info, best_match_info['score'], original_scene_cv_bgr, patch_gen_time, matching_time
    else:
        return None, max_score_overall, original_scene_cv_bgr, patch_gen_time, matching_time

# Approach 2: SAM + DINOv2 Matching (more efficient and more accurate)
def match_with_sam_dinov2(query_embedding_data, scene_image_path,
                          similarity_threshold=0.75,
                          batch_size=16, max_sam_dim=512):
    if not sam_loaded:
        try: scene_cv_bgr = cv2.imread(scene_image_path)
        except: scene_cv_bgr = np.zeros((100,100,3), dtype=np.uint8)
        return None, -1.0, scene_cv_bgr if scene_cv_bgr is not None else np.zeros((100,100,3), dtype=np.uint8), 0.0, 0.0

    query_embedding, _, _ = query_embedding_data # Unpack query data, w/h not used here directly

    scene_cv_bgr_original = cv2.imread(scene_image_path)
    if scene_cv_bgr_original is None:
        return None, -1.0, np.zeros((100,100,3), dtype=np.uint8), 0.0, 0.0

    h_orig, w_orig = scene_cv_bgr_original.shape[:2]
    scale_factor = 1.0
    if max(h_orig, w_orig) > max_sam_dim:
        if h_orig > w_orig:
            scale_factor = max_sam_dim / h_orig
            new_h = max_sam_dim
            new_w = int(w_orig * scale_factor)
        else:
            scale_factor = max_sam_dim / w_orig
            new_w = max_sam_dim
            new_h = int(h_orig * scale_factor)
        print(f"\tResizing scene for SAM from {w_orig}x{h_orig} to {new_w}x{new_h}")
        scene_cv_bgr_for_sam = cv2.resize(scene_cv_bgr_original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scene_cv_rgb_np = cv2.cvtColor(scene_cv_bgr_for_sam, cv2.COLOR_BGR2RGB)
    else:
        scene_cv_rgb_np = cv2.cvtColor(scene_cv_bgr_original, cv2.COLOR_BGR2RGB)

    time_sam_gen_start = time.time()
    print("\tGenerating masks with SAM...")
    sam_masks_data = mask_generator.generate(scene_cv_rgb_np)
    time_sam_gen_end = time.time()
    sam_gen_time = time_sam_gen_end - time_sam_gen_start
    print(f"\tSAM generated {len(sam_masks_data)} masks in {sam_gen_time:.2f}s.")

    if not sam_masks_data: return None, -1.0, scene_cv_bgr_original, sam_gen_time, 0.0

    segments_to_embed_pil = []
    segment_metadata_list = []
    for original_idx, mask_data in enumerate(sam_masks_data):
        mask_np = mask_data['segmentation']
        bbox_xywh_sam = mask_data['bbox'] 
        x_s, y_s, w_s, h_s = bbox_xywh_sam
        if w_s < 10 or h_s < 10: continue

        cropped_rgb_np = scene_cv_rgb_np[y_s:y_s+h_s, x_s:x_s+w_s]
        cropped_mask_np = mask_np[y_s:y_s+h_s, x_s:x_s+w_s]
        
        masked_segment_np = np.zeros_like(cropped_rgb_np)
        masked_segment_np[cropped_mask_np] = cropped_rgb_np[cropped_mask_np]
        
        segment_pil = Image.fromarray(masked_segment_np)
        segments_to_embed_pil.append(segment_pil)
        
        x_orig = int(x_s / scale_factor)
        y_orig = int(y_s / scale_factor)
        w_orig_bbox = int(w_s / scale_factor)
        h_orig_bbox = int(h_s / scale_factor)
        bbox_xywh_orig = (x_orig, y_orig, w_orig_bbox, h_orig_bbox)
        segment_metadata_list.append({'bbox_orig': bbox_xywh_orig, 'original_mask_idx': original_idx})

    if not segments_to_embed_pil:
        return None, -1.0, scene_cv_bgr_original, sam_gen_time, 0.0

    best_match_score_overall = -1.0
    best_match_info_passing_thresh = None
    
    time_matching_start = time.time()
    for i in range(0, len(segments_to_embed_pil), batch_size):
        batch_segment_pils = segments_to_embed_pil[i:i+batch_size]
        batch_segment_metadata = segment_metadata_list[i:i+batch_size]
        if not batch_segment_pils: continue
        
        segment_batch_pixel_values, _ = preprocess_for_dino(batch_segment_pils, dino_image_processor)
        segment_embeddings = get_dino_embedding(segment_batch_pixel_values, dino_model)
        similarities = cosine_similarity(query_embedding, segment_embeddings)[0]
        
        for j, score_float in enumerate(similarities):
            score = float(score_float)
            if score > best_match_score_overall:
                best_match_score_overall = score
            
            if score >= similarity_threshold:
                if best_match_info_passing_thresh is None or score > best_match_info_passing_thresh['score']:
                    current_metadata = batch_segment_metadata[j]
                    best_match_info_passing_thresh = {
                        'coords': current_metadata['bbox_orig'],
                        'score': score
                    }
    time_matching_end = time.time()
    matching_time = time_matching_end - time_matching_start
    
    if best_match_info_passing_thresh:
        return best_match_info_passing_thresh, best_match_info_passing_thresh['score'], scene_cv_bgr_original, sam_gen_time, matching_time
    else:
        return None, best_match_score_overall, scene_cv_bgr_original, sam_gen_time, matching_time


# MAIN
if __name__ == '__main__':
    if not os.path.isdir(SINGLES_DIR):
        print(f"ERROR: Singles directory not found: {SINGLES_DIR}")
        exit()
    if not os.path.isdir(COMPOSITES_DIR):
        print(f"ERROR: Composites directory not found: {COMPOSITES_DIR}")
        exit()

    query_files = sorted([f for f in os.listdir(SINGLES_DIR) if os.path.isfile(os.path.join(SINGLES_DIR, f))])
    composite_files = sorted([f for f in os.listdir(COMPOSITES_DIR) if os.path.isfile(os.path.join(COMPOSITES_DIR, f))])

    overall_start_time = time.time()

    for query_idx, query_filename in enumerate(query_files):
        query_image_path = os.path.join(SINGLES_DIR, query_filename)
        query_name_base = os.path.splitext(query_filename)[0]

        current_query_results_dir = os.path.join(BASE_RESULTS_DIR, query_name_base)
        os.makedirs(current_query_results_dir, exist_ok=True)
        
        results_log_path = os.path.join(current_query_results_dir, "results_summary.txt")

        print(f"\n--- Processing Query {query_idx+1}/{len(query_files)}: {query_filename} ---")
        
        time_query_embed_start = time.time()
        query_pixel_values, query_pil_for_size = preprocess_for_dino(query_image_path, dino_image_processor)
        query_embedding = get_dino_embedding(query_pixel_values, dino_model)
        q_w, q_h = query_pil_for_size.size
        query_embedding_data = (query_embedding, q_w, q_h) # Package for patch matcher
        time_query_embed_end = time.time()
        query_embed_time = time_query_embed_end - time_query_embed_start
        print(f"\tQuery embedding generated in {query_embed_time:.2f}s")

        with open(results_log_path, "w") as log_file:
            log_file.write(f"Results for Query: {query_filename}\n")
            log_file.write(f"Query Embedding Time: {query_embed_time:.4f} seconds\n")
            log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("="*40 + "\n")

            for scene_idx, scene_filename in enumerate(composite_files):
                scene_image_path = os.path.join(COMPOSITES_DIR, scene_filename)
                scene_name_base = os.path.splitext(scene_filename)[0]

                print(f"  Matching against Scene {scene_idx+1}/{len(composite_files)}: {scene_filename}")
                log_file.write(f"\nScene: {scene_filename}\n")
                
                pair_processing_start_time = time.time()
                match_info = None
                best_score_achieved = -1.0
                result_image_cv = None
                method_used = ""
                method_specific_time1 = 0.0 # = SAM gen time or Patch gen time
                method_specific_time2 = 0.0 # = Matching time

                if sam_loaded:
                    method_used = "SAM+DINOv2"
                    match_info, best_score_achieved, original_scene_cv, method_specific_time1, method_specific_time2 = match_with_sam_dinov2(
                        query_embedding_data, scene_image_path,
                        similarity_threshold=0.25,
                        max_sam_dim=512
                    )
                    result_image_cv = draw_match_info_on_image(original_scene_cv, match_info, color=(0,0,255))
                    log_file.write(f"    SAM Mask Generation Time: {method_specific_time1:.4f}s\n")
                    log_file.write(f"    Segment Matching Time: {method_specific_time2:.4f}s\n")
                else:
                    method_used = "DINOv2 Patch"
                    match_info, best_score_achieved, original_scene_cv, method_specific_time1, method_specific_time2 = match_with_dinov2_patches(
                        query_embedding_data, scene_image_path,
                        similarity_threshold=0.50,
                        nms_iou_threshold=0.15,
                        patch_scales=[0.7, 1.0, 1.3]
                    )
                    result_image_cv = draw_match_info_on_image(original_scene_cv, match_info, color=(0,255,0))
                    log_file.write(f"    Patch Generation Time: {method_specific_time1:.4f}s\n")
                    log_file.write(f"    Patch Matching Time: {method_specific_time2:.4f}s\n")
                
                pair_processing_end_time = time.time()
                pair_total_time = pair_processing_end_time - pair_processing_start_time

                log_file.write(f"  Method Used: {method_used}\n")
                if match_info:
                    log_file.write(f"  Match Found: True\n")
                    log_file.write(f"  Match Score: {match_info['score']:.4f}\n")
                    log_file.write(f"  Match Coordinates (x,y,w,h): {match_info['coords']}\n")
                    print(f"\t  Match Found! Score: {match_info['score']:.3f} ({method_used})")
                else:
                    log_file.write(f"  Match Found: False\n")
                    log_file.write(f"  Best Score Achieved (below threshold): {best_score_achieved:.4f}\n")
                    print(f"\t  No confident match. Best score: {best_score_achieved:.3f} ({method_used})")
                
                log_file.write(f"  Total Scene Processing Time: {pair_total_time:.4f} seconds\n")

                output_image_filename = f"{method_used.replace('+', '_').replace(' ', '_')}_{scene_name_base}_result.png"
                output_image_path = os.path.join(current_query_results_dir, output_image_filename)
                
                if result_image_cv is not None:
                    cv2.imwrite(output_image_path, result_image_cv)
                    log_file.write(f"  Saved Image: {output_image_filename}\n")
                
                # Attempt to free memory
                del match_info, result_image_cv, original_scene_cv
                gc.collect()

            log_file.write("="*40)
        print(f"  Finished processing for query {query_filename}. Results in: {current_query_results_dir}")
        del query_embedding, query_pixel_values, query_pil_for_size, query_embedding_data
        gc.collect()

    overall_end_time = time.time()
    total_script_time = overall_end_time - overall_start_time
    print(f"\nAll queries processed in {total_script_time:.2f} seconds. Main results directory: {os.path.abspath(BASE_RESULTS_DIR)}")