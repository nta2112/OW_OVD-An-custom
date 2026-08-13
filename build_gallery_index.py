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

    # 2. Patch version limits BEFORE importing mmdet/mmyolo to prevent AssertionError
    import importlib.util
    import re
    def _patch_file(pkg_name, new_ver="2.3.0"):
        try:
            spec = importlib.util.find_spec(pkg_name)
            if spec is not None and spec.origin:
                init_file = spec.origin
                with open(init_file, "r", encoding="utf-8") as f:
                    txt = f.read()
                pattern = r"(mmcv_maximum_version\s*=\s*['\"])([^'\"]+)(['\"])"
                new_txt, count = re.subn(pattern, rf"\g<1>{new_ver}\g<3>", txt)
                if count > 0 and new_txt != txt:
                    with open(init_file, "w", encoding="utf-8") as f:
                        f.write(new_txt)
                    print(f"-> Patched {count} MMCV version limit(s) to {new_ver} in {pkg_name}")
                    pycache = os.path.join(os.path.dirname(init_file), "__pycache__")
                    if os.path.exists(pycache):
                        import shutil
                        try:
                            shutil.rmtree(pycache)
                        except Exception:
                            pass
        except Exception:
            pass

    _patch_file("mmdet")
    _patch_file("mmyolo")

    # 3. Now it is safe to import mmengine and mmyolo
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

    # 2. Collect images (Case-insensitive & Annotation-driven fallback)
    image_paths = []
    resolved_dir = args.gallery_dir

    # First, let's resolve the folder containing the physical image files
    found_folder = False
    if os.path.exists(resolved_dir):
        for root, _, files in os.walk(resolved_dir):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                resolved_dir = root
                found_folder = True
                break
                
    if not found_folder:
        parent_dir = os.path.dirname(args.gallery_dir)
        print(f"-> WARNING: Specified directory empty. Scanning parent {parent_dir} recursively...")
        for root, _, files in os.walk(parent_dir):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                resolved_dir = root
                found_folder = True
                break
                
    # Now, if an annotation file is provided, filter the images to include only those in the split
    if args.ann_file and os.path.exists(args.ann_file):
        print(f"-> Filtering gallery images using split annotations: {args.ann_file}")
        with open(args.ann_file, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        for img in coco.get('images', []):
            basename = os.path.basename(img['file_name'])
            real_path = os.path.join(resolved_dir, basename)
            if os.path.exists(real_path):
                image_paths.append(real_path)
            else:
                import glob
                matches = glob.glob(os.path.join(resolved_dir, '**', basename), recursive=True)
                if matches:
                    image_paths.append(matches[0])
        print(f"-> Selected {len(image_paths)} images belonging to split defined in {args.ann_file}")
    else:
        # Otherwise, collect all images in the directory
        if found_folder:
            for root, _, files in os.walk(resolved_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, file))
                        
    if not image_paths:
        print(f"Error: No images found in {resolved_dir} or parent directory.")
        return
        
    print(f"-> Total images to index: {len(image_paths)}")

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
    
    for img_path in tqdm(image_paths, desc="Processing Gallery Images"):
        filename = os.path.basename(img_path)
        
        # Determine class info
        class_id = filename_to_class_id.get(filename, -1)
        if class_id != -1:
            class_label = cat_id_to_name.get(class_id, f"class_{class_id}")
        else:
            # Fallback: parent directory name
            class_label = os.path.basename(os.path.dirname(img_path))
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
                # Unpack if returned as BaseModelOutputWithPooling
                if not isinstance(features, torch.Tensor):
                    if hasattr(features, "pooler_output") and features.pooler_output is not None:
                        features = features.pooler_output
                    elif hasattr(features, "image_embeds") and features.image_embeds is not None:
                        features = features.image_embeds
                    elif isinstance(features, (list, tuple)):
                        features = features[0]
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
