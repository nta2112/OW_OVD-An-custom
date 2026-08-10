"""
Dataset-Wide Retrieval Evaluation
=================================
This script evaluates the Two-Stage Crop-then-Search Image Retrieval pipeline
across the entire dataset split. It calculates standard retrieval metrics:
Recall@1, Recall@5, and Recall@10 (both class-wise and global averages).

To avoid redundant compute, it supports saving/loading pre-computed embedding caches.

Usage:
    python evaluate_retrieval.py \
        --config configs/open_world/mowod/custom/ip102_t1.py \
        --checkpoint pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth \
        --dataset-root data/IP102 \
        --query-split val \
        --gallery-split test \
        --output-report evaluation_report.md
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
from typing import List, Dict, Tuple

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
    parser = argparse.ArgumentParser(description="Evaluate CBIR Pest Retrieval System")
    parser.add_argument("--config", type=str, default="configs/open_world/mowod/custom/ip102_t1.py",
                        help="Path to detector config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to detector checkpoint file")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to IP102 dataset root directory")
    parser.add_argument("--query-split", type=str, default="val",
                        help="Dataset split to use as Query (e.g. val, test)")
    parser.add_argument("--gallery-split", type=str, default="test",
                        help="Dataset split to use as Gallery (e.g. test, train)")
    parser.add_argument("--query-cache", type=str, default="query_cache.pkl",
                        help="Path to save/load query embeddings cache")
    parser.add_argument("--gallery-cache", type=str, default="gallery_cache.pkl",
                        help="Path to save/load gallery embeddings cache")
    parser.add_argument("--output-report", type=str, default="retrieval_evaluation_report.md",
                        help="Path to save markdown report file")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP vision model to use")
    parser.add_argument("--score-thr", type=float, default=0.35,
                        help="Confidence threshold for pest detection")
    return parser.parse_args()


def load_dataset_records(dataset_root: str, split: str) -> List[Dict]:
    """
    Parses COCO JSON and collects image files and their categories.
    """
    json_path = os.path.join(dataset_root, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON annotation file not found: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
        
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    
    # Image folder fallback search
    image_folder = dataset_root
    found_folder = False
    for subfolder in [split, 'images', 'test/test', 'train/train', 'val/val']:
        test_path = os.path.join(dataset_root, subfolder)
        if os.path.exists(test_path) and os.path.isdir(test_path):
            try:
                files_in_dir = os.listdir(test_path)
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files_in_dir):
                    image_folder = test_path
                    found_folder = True
                    break
            except Exception:
                continue
                
    if not found_folder:
        # Recursive search for first folder containing images
        print(f"-> WARNING: Standard subfolders empty. Scanning {dataset_root} recursively to locate image files...")
        for root, _, files in os.walk(dataset_root):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                image_folder = root
                found_folder = True
                print(f"-> Auto-resolved image directory to: {image_folder}")
                break
            
    images = coco.get('images', [])
    img_id_to_info = {}
    for img in images:
        file_name = os.path.basename(img['file_name'])
        img_id_to_info[img['id']] = {
            'file_name': file_name,
            'image_path': os.path.join(image_folder, file_name)
        }
        
    annotations = coco.get('annotations', [])
    img_to_class = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_class:
            img_to_class[img_id] = ann['category_id']
            
    records = []
    for img_id, info in img_id_to_info.items():
        class_id = img_to_class.get(img_id, -1)
        if class_id == -1:
            continue
            
        real_path = info['image_path']
        # Try finding file if missing
        if not os.path.exists(real_path):
            import glob
            basename = os.path.basename(real_path)
            matches = glob.glob(os.path.join(dataset_root, '**', basename), recursive=True)
            if matches:
                real_path = matches[0]
            else:
                continue
                
        records.append({
            "image_path": real_path,
            "class_id": class_id,
            "class_label": cat_id_to_name.get(class_id, f"class_{class_id}")
        })
        
    return records


def extract_split_embeddings(
    records: List[Dict], 
    model, 
    clip_model, 
    clip_processor, 
    device: str, 
    score_thr: float,
    desc: str
) -> List[Dict]:
    """
    Extracts visual feature embeddings for a list of images. Performs crop-then-search.
    """
    processed_records = []
    
    for item in tqdm(records, desc=desc):
        try:
            img_pil = Image.open(item['image_path']).convert("RGB")
            width, height = img_pil.size
            
            # Detect
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            det_result = inference_detector(model, img_bgr)
            pred_instances = det_result.pred_instances
            boxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            
            best_box = None
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                if scores[best_idx] >= score_thr:
                    best_box = boxes[best_idx]
                    
            # Stage 1: Crop with 10% padding
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                box_w = x2 - x1
                box_h = y2 - y1
                pad_w = int(0.1 * box_w)
                pad_h = int(0.1 * box_h)
                
                x1_pad = max(0, int(x1 - pad_w))
                y1_pad = max(0, int(y1 - pad_h))
                x2_pad = min(width, int(x2 + pad_w))
                y2_pad = min(height, int(y2 + pad_h))
                
                cropped_img = img_pil.crop((x1_pad, y1_pad, x2_pad, y2_pad))
            else:
                cropped_img = img_pil
                
            # Stage 2: Feature extraction
            inputs = clip_processor(images=cropped_img, return_tensors="pt").to(device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                feature_vector = features.cpu().numpy()[0]
                
            processed_records.append({
                "image_path": item['image_path'],
                "class_id": item['class_id'],
                "class_label": item['class_label'],
                "feature_vector": feature_vector
            })
        except Exception as e:
            continue
            
    return processed_records


def evaluate_retrieval(query_data: List[Dict], gallery_data: List[Dict]) -> Tuple[Dict, float, float, float]:
    """
    Computes Recall@1, Recall@5, and Recall@10 metrics.
    """
    gallery_embeddings = np.array([item["feature_vector"] for item in gallery_data])
    gallery_classes = [item["class_id"] for item in gallery_data]
    
    class_eval = {}
    
    for q_item in tqdm(query_data, desc="Matching Queries"):
        q_class = q_item["class_id"]
        q_label = q_item["class_label"]
        q_embed = q_item["feature_vector"]
        
        # Compute Cosine Similarity
        sims = np.dot(q_embed, gallery_embeddings.T)
        
        # Sort indices
        top_indices = np.argsort(sims)[::-1][:10]
        top_classes = [gallery_classes[idx] for idx in top_indices]
        
        if q_label not in class_eval:
            class_eval[q_label] = {1: [], 5: [], 10: []}
            
        # Recall@K check
        rec1 = 1.0 if q_class in top_classes[:1] else 0.0
        rec5 = 1.0 if q_class in top_classes[:5] else 0.0
        rec10 = 1.0 if q_class in top_classes[:10] else 0.0
        
        class_eval[q_label][1].append(rec1)
        class_eval[q_label][5].append(rec5)
        class_eval[q_label][10].append(rec10)
        
    class_metrics = {}
    macro_r1, macro_r5, macro_r10 = 0.0, 0.0, 0.0
    weighted_r1, weighted_r5, weighted_r10 = 0.0, 0.0, 0.0
    total_queries = 0
    
    for label, data in class_eval.items():
        count = len(data[1])
        r1 = np.mean(data[1])
        r5 = np.mean(data[5])
        r10 = np.mean(data[10])
        
        class_metrics[label] = {
            "count": count,
            "R@1": r1,
            "R@5": r5,
            "R@10": r10
        }
        
        macro_r1 += r1
        macro_r5 += r5
        macro_r10 += r10
        
        weighted_r1 += r1 * count
        weighted_r5 += r5 * count
        weighted_r10 += r10 * count
        total_queries += count
        
    num_classes = len(class_metrics)
    if num_classes > 0:
        macro_r1 /= num_classes
        macro_r5 /= num_classes
        macro_r10 /= num_classes
        
        weighted_r1 /= total_queries
        weighted_r5 /= total_queries
        weighted_r10 /= total_queries
        
    return class_metrics, macro_r1, macro_r5, macro_r10


def save_report(
    report_path: str, 
    class_metrics: Dict, 
    macro_r1: float, 
    macro_r5: float, 
    macro_r10: float,
    query_split: str,
    gallery_split: str
):
    """
    Saves a beautifully formatted Markdown report file.
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Agricultural Pest Retrieval System Evaluation Report\n\n")
        f.write(f"- **Query Split:** `{query_split}`\n")
        f.write(f"- **Gallery Split:** `{gallery_split}`\n\n")
        
        f.write("## Summary Metrics\n\n")
        f.write("| Metric | Macro Average | Weighted Average |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Recall@1** | {macro_r1:.4f} | - |\n")
        f.write(f"| **Recall@5** | {macro_r5:.4f} | - |\n")
        f.write(f"| **Recall@10** | {macro_r10:.4f} | - |\n\n")
        
        f.write("## Class-Wise Retrieval Metrics\n\n")
        f.write("| Class Name | Query Count | Recall@1 | Recall@5 | Recall@10 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls_name, metrics in sorted(class_metrics.items()):
            f.write(f"| {cls_name} | {metrics['count']} | {metrics['R@1']:.4f} | {metrics['R@5']:.4f} | {metrics['R@10']:.4f} |\n")
            
    print(f"-> Saved markdown evaluation report to: {report_path}")


def main():
    args = parse_args()
    print("="*60)
    print("      IP102 RETRIEVAL METRICS EVALUATION PIPELINE      ")
    print("="*60)
    
    # 1. Load Dataset Splits Info
    print("-> Reading annotations...")
    query_records = load_dataset_records(args.dataset_root, args.query_split)
    gallery_records = load_dataset_records(args.dataset_root, args.gallery_split)
    print(f"-> Found {len(query_records)} query images and {len(gallery_records)} gallery images.")

    # 2. Extract Query features (or load cache)
    if os.path.exists(args.query_cache):
        print(f"-> Loading pre-computed Query embeddings from cache: {args.query_cache}")
        with open(args.query_cache, "rb") as f:
            query_processed = pickle.load(f)
    else:
        print("-> Query cache not found. Extracting embeddings on the fly...")
        model = init_detector(args.config, args.checkpoint, device=args.device)
        clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(args.device)
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
        model.eval()
        clip_model.eval()
        
        query_processed = extract_split_embeddings(
            query_records, model, clip_model, clip_processor, args.device, args.score_thr, "Processing Query Split"
        )
        with open(args.query_cache, "wb") as f:
            pickle.dump(query_processed, f)
        print(f"-> Saved Query cache to {args.query_cache}")

    # 3. Extract Gallery features (or load cache)
    if os.path.exists(args.gallery_cache):
        print(f"-> Loading pre-computed Gallery embeddings from cache: {args.gallery_cache}")
        with open(args.gallery_cache, "rb") as f:
            gallery_processed = pickle.load(f)
    else:
        print("-> Gallery cache not found. Extracting embeddings on the fly...")
        # Load models if not already loaded
        if 'model' not in locals():
            model = init_detector(args.config, args.checkpoint, device=args.device)
            clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(args.device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
            model.eval()
            clip_model.eval()
            
        gallery_processed = extract_split_embeddings(
            gallery_records, model, clip_model, clip_processor, args.device, args.score_thr, "Processing Gallery Split"
        )
        with open(args.gallery_cache, "wb") as f:
            pickle.dump(gallery_processed, f)
        print(f"-> Saved Gallery cache to {args.gallery_cache}")

    # 4. Perform Retrieval Evaluation
    print("-> Calculating retrieval metrics...")
    metrics, r1, r5, r10 = evaluate_retrieval(query_processed, gallery_processed)
    
    # 5. Display Summary
    print("\n" + "="*50)
    print("            SUMMARY RETRIEVAL METRICS            ")
    print("="*50)
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print("="*50)
    
    # 6. Save Markdown Report
    save_report(args.output_report, metrics, r1, r5, r10, args.query_split, args.gallery_split)
    print("====== Evaluation Process Completed Successfully! ======")


if __name__ == "__main__":
    main()
