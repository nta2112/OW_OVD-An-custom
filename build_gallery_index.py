"""
Offline Gallery Database Builder
================================
This script processes a gallery folder of pest images to build an embedding index.
It runs the OW-OVD detector (YOLO-World) on each image, crops the detected pest
region (with 10% padding and boundary clamping), extracts its visual embedding
using CLIP, and saves the index database.

Usage:
    python build_gallery_index.py \
        --config configs/open_world/mowod/custom/ip102_t1.py \
        --checkpoint pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth \
        --gallery-dir data/IP102/test \
        --ann-file data/IP102/test.json \
        --output gallery_index.pkl \
        --device cuda:0
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Lazy patch to ensure compatibility with MMCV version limits and pure Python fallback
def patch_environment():
    try:
        import mmcv._ext
    except ImportError:
        import types
        import importlib.machinery
        if 'mmcv._ext' not in sys.modules:
            class MockModule(types.ModuleType):
                def __getattr__(self, name):
                    if name.startswith('__'):
                        raise AttributeError(name)
                    if name == 'nms':
                        import torchvision.ops as tv_ops
                        return tv_ops.nms
                    def dummy_func(*args, **kwargs):
                        raise NotImplementedError(f"C++ operation '{name}' not supported on CPU fallback.")
                    return dummy_func
            mock_ext = MockModule('mmcv._ext')
            mock_ext.__spec__ = importlib.machinery.ModuleSpec('mmcv._ext', None)
            sys.modules['mmcv._ext'] = mock_ext

    import mmengine
    from mmyolo import __file__ as mmyolo_init_path
    mmyolo_pkg_root = os.path.dirname(mmyolo_init_path)
    
    _orig_file2dict = mmengine.Config._file2dict
    @classmethod
    def _patched_file2dict(cls, filename, *args, **kwargs):
        filename_str = str(filename).replace('\\', '/')
        if 'third_party/mmyolo/configs' in filename_str:
            relative_part = filename_str.split('third_party/mmyolo/configs/')[-1]
            new_path = os.path.join(mmyolo_pkg_root, '.mim', 'configs', relative_part)
            if os.path.exists(new_path):
                filename = new_path
            else:
                new_path_fallback = os.path.join(mmyolo_pkg_root, 'configs', relative_part)
                if os.path.exists(new_path_fallback):
                    filename = new_path_fallback
        return _orig_file2dict(filename, *args, **kwargs)
    mmengine.Config._file2dict = _patched_file2dict

    import importlib.util
    def _patch(pkg, old, new="2.3.0"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.origin: return
        with open(spec.origin, "r", encoding="utf-8") as f:
            txt = f.read()
        old_str = f"mmcv_maximum_version = '{old}'"
        if old_str in txt:
            with open(spec.origin, "w", encoding="utf-8") as f:
                f.write(txt.replace(old_str, f"mmcv_maximum_version = '{new}'"))
    _patch("mmdet", "2.2.0")
    _patch("mmdet", "2.1.0")
    _patch("mmyolo", "2.1.0")
    _patch("mmyolo", "2.2.0")


# Run patches immediately
patch_environment()

import torch
import cv2
from mmdet.apis import init_detector, inference_detector
from transformers import CLIPProcessor, CLIPModel


def parse_args():
    parser = argparse.ArgumentParser(description="Build offline gallery database index")
    parser.add_argument("--config", type=str, default="configs/open_world/mowod/custom/ip102_t1.py",
                        help="Path to detector config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to detector checkpoint file")
    parser.add_argument("--gallery-dir", type=str, required=True,
                        help="Path to directory containing gallery images")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="Optional COCO JSON annotation file to map images to class labels")
    parser.add_argument("--output", type=str, default="gallery_index.pkl",
                        help="Path to output pickle file")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP vision model to use")
    parser.add_argument("--score-thr", type=float, default=0.35,
                        help="Confidence threshold for pest detection")
    return parser.parse_args()


def load_annotations(ann_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Parses COCO JSON file and maps filename to category_id and category_name.
    """
    if not ann_file or not os.path.exists(ann_file):
        return {}, {}
        
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
        
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    
    # Map image ID to filename
    img_id_to_filename = {}
    for img in coco.get('images', []):
        img_id_to_filename[img['id']] = os.path.basename(img['file_name'])
        
    # Map image filename to category_id (take the first annotation class as label)
    filename_to_class_id = {}
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        if img_id in img_id_to_filename:
            fname = img_id_to_filename[img_id]
            if fname not in filename_to_class_id:
                filename_to_class_id[fname] = ann['category_id']
                
    return filename_to_class_id, cat_id_to_name


def main():
    args = parse_args()
    print("="*60)
    print("      IP102 CBIR GALLERY INDEX BUILDER (OFFLINE)      ")
    print("="*60)
    
    # 1. Load COCO annotations if provided
    filename_to_class_id, cat_id_to_name = load_annotations(args.ann_file)
    if filename_to_class_id:
        print(f"-> Loaded annotations mapping for {len(filename_to_class_id)} images.")
    else:
        print("-> No annotation mapping loaded. Filenames/folders will be used as fallback labels.")

    # 2. Collect images in directory (Case-insensitive & Auto-resolving)
    image_paths = []
    resolved_dir = args.gallery_dir
    
    # Try finding images in the specified directory
    if os.path.exists(resolved_dir):
        for root, _, files in os.walk(resolved_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
                    
    # Auto-resolve if empty by searching the dataset root recursively
    if len(image_paths) == 0:
        parent_dir = os.path.dirname(args.gallery_dir)
        print(f"-> WARNING: No images found in {args.gallery_dir}. Searching dataset parent directory recursively...")
        for root, _, files in os.walk(parent_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
                    resolved_dir = root # update resolved folder to first folder containing images
        if image_paths:
            print(f"-> Auto-resolved image directory to: {resolved_dir}")
         
    if not image_paths:
        print(f"Error: No images found in {args.gallery_dir} or parent {os.path.dirname(args.gallery_dir)}")
        return
        
    print(f"-> Found {len(image_paths)} images in gallery directory: {resolved_dir}")

    # 3. Load OW-OVD Detector Model
    print(f"-> Initializing Detector model on {args.device}...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    # 4. Load CLIP model
    print(f"-> Loading CLIP model {args.clip_model}...")
    clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model.eval()

    # 5. Process images
    gallery_records = []
    
    for img_path_obj in tqdm(image_paths, desc="Processing Gallery Images"):
        img_path = str(img_path_obj)
        filename = img_path_obj.name
        
        # Determine class info
        class_id = filename_to_class_id.get(filename, -1)
        if class_id != -1:
            class_label = cat_id_to_name.get(class_id, f"class_{class_id}")
        else:
            # Fallback: parent directory name
            class_label = img_path_obj.parent.name
            class_id = -1
            
        try:
            # Load image
            img_pil = Image.open(img_path).convert("RGB")
            width, height = img_pil.size
            
            # Run OW-OVD Detector
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            det_result = inference_detector(model, img_bgr)
            
            pred_instances = det_result.pred_instances
            boxes = pred_instances.bboxes.cpu().numpy()  # (N, 4) xyxy
            scores = pred_instances.scores.cpu().numpy()  # (N,)
            
            # Find the best pest box
            best_box = None
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                if scores[best_idx] >= args.score_thr:
                    best_box = boxes[best_idx]
            
            # Stage 1: Crop region of interest with 10% padding
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Apply 10% padding
                pad_w = int(0.1 * box_w)
                pad_h = int(0.1 * box_h)
                
                x1_pad = max(0, int(x1 - pad_w))
                y1_pad = max(0, int(y1 - pad_h))
                x2_pad = min(width, int(x2 + pad_w))
                y2_pad = min(height, int(y2 + pad_h))
                
                # Crop
                cropped_img = img_pil.crop((x1_pad, y1_pad, x2_pad, y2_pad))
            else:
                # No Detection Fallback: use whole image
                cropped_img = img_pil

            # Stage 2: Extract CLIP feature embeddings
            inputs = clip_processor(images=cropped_img, return_tensors="pt").to(args.device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                # L2 Normalize
                features = features / features.norm(dim=-1, keepdim=True)
                feature_vector = features.cpu().numpy()[0]
                
            gallery_records.append({
                "image_path": img_path,
                "class_id": class_id,
                "class_label": class_label,
                "feature_vector": feature_vector
            })
            
        except Exception as e:
            print(f"\nError processing image {img_path}: {e}")
            continue

    # 6. Save gallery index
    print(f"-> Saving gallery index with {len(gallery_records)} records to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(gallery_records, f)
        
    print("====== Gallery Database Indexing Completed Successfully! ======")


if __name__ == "__main__":
    main()
