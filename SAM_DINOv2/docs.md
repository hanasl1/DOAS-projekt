# About

Script to run SAM region detection + DINOv2 embedding (representation) on each found ROI, then compare the embeddings to the embedding of the query (single object) image

# Setup

1. Download vit_b SAM model from https://github.com/facebookresearch/segment-anything#model-checkpoints
2. Run:
```bash
pip install -r requirements.txt
```
3. Prepare dataset of images of individual objects + scenes

# Usage

Run:
```bash
python main_sam_dino.py
```