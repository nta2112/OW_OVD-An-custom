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
import torch.nn.functional as F
import cv2
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
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
    default_clip = "openai/clip-vit-base-patch32"
    local_clip_path = "/kaggle/input/models/yujkaggle/openaiclip-vit-base-patch32/pytorch/default/1"
    if os.path.exists(local_clip_path):
        default_clip = local_clip_path
        print(f"-> Detected offline Kaggle CLIP model. Defaulting to: {local_clip_path}")
        
    parser.add_argument("--clip-model", type=str, default=default_clip,
                        help="CLIP vision model to use")
    parser.add_argument("--score-thr", type=float, default=0.35,
                        help="Confidence threshold for pest detection")
    parser.add_argument("--detector-retrieval", action="store_true",
                        help="Use the detector's own retrieval head instead of CLIP")
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


def extract_hauf_safeguard(model, crop_img: Image.Image, device: str) -> float:
    """
    Passes the cropped image through the detector model and calculates the 
    HAUF unknown anomaly score (P_u) directly from the model's logits.
    """
    try:
        # Preprocess crop image to BGR numpy array
        crop_cv = cv2.cvtColor(np.array(crop_img.resize((640, 640))), cv2.COLOR_RGB2BGR)
        crop_tensor = torch.from_numpy(crop_cv).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Prepare data sample
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': crop_cv.shape[:2],
            'ori_shape': crop_cv.shape[:2],
            'pad_shape': crop_cv.shape[:2],
        })
        
        # Normalization and padding
        with torch.no_grad():
            batch_inputs, batch_data_samples = model.data_preprocessor(crop_tensor, [data_sample])
            
            # Extract features
            img_feats, txt_feats = model.extract_feat(batch_inputs, batch_data_samples)
            raw_outs = model.bbox_head(img_feats, txt_feats)
            
            # Append unknown predictions using model's head
            att_feats = model.bbox_head.att_embeddings[None].repeat(img_feats[0].shape[0], 1, 1)
            cls_scores_with_unknown, _ = model.bbox_head.predict_unknown(raw_outs, img_feats, att_feats)
            
            # Find the highest unknown score (P_u) across all levels and grid cells
            max_pu = -1.0
            
            for i, logits in enumerate(cls_scores_with_unknown):
                pu_map = logits[0, -1]  # The last channel is P_u (already sigmoid-activated)
                val, idx = pu_map.view(-1).max(dim=0)
                if val.item() > max_pu:
                    max_pu = val.item()
                    
            return max_pu
            
    except Exception as e:
        return 0.0


def calculate_auroc_numpy(y_true, y_scores) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the Area Under the ROC Curve (AUROC) using numpy without sklearn.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_score_indices]
    y_scores = y_scores[desc_score_indices]
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])
        
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    
    tpr = tp / n_pos
    fpr = fp / n_neg
    
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    
    # Trapezoidal integration
    area = 0.0
    for i in range(len(fpr) - 1):
        area += 0.5 * (tpr[i] + tpr[i+1]) * (fpr[i+1] - fpr[i])
        
    return area, fpr, tpr


def calculate_fpr_at_tpr95(fpr, tpr) -> float:
    """
    Finds the FPR when TPR is >= 0.95.
    """
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    return 1.0


def extract_split_embeddings(
    records: List[Dict], 
    model, 
    clip_model, 
    clip_processor, 
    device: str, 
    score_thr: float,
    desc: str,
    use_detector_retrieval: bool = False,
    batch_size: int = 64
) -> List[Dict]:
    """
    Extracts visual feature embeddings for a list of images. Performs crop-then-search.
    """
    processed_records = []
    crops = []
    valid_indices = []
    
    # Bước 1: Phát hiện đối tượng và cắt ảnh (Detection & Crop)
    for idx, item in enumerate(tqdm(records, desc=f"{desc} (BBox Detection)")):
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
            best_idx = -1
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                if scores[best_idx] >= score_thr:
                    best_box = boxes[best_idx]
                    
            # Stage 1: Crop with 10% padding (only if using CLIP)
            cropped_img = None
            if not use_detector_retrieval:
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
                
            # Compute HAUF Anomaly score
            unknown_score = 0.0
            if best_box is not None and hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'att_embeddings') and model.bbox_head.att_embeddings is not None:
                # Use cropped_img for HAUF if CLIP, otherwise crop from img_pil
                if cropped_img is None:
                    x1, y1, x2, y2 = best_box
                    x1_pad = max(0, int(x1 - 0.1*(x2-x1)))
                    y1_pad = max(0, int(y1 - 0.1*(y2-y1)))
                    x2_pad = min(width, int(x2 + 0.1*(x2-x1)))
                    y2_pad = min(height, int(y2 + 0.1*(y2-y1)))
                    cropped_hauf_img = img_pil.crop((x1_pad, y1_pad, x2_pad, y2_pad))
                else:
                    cropped_hauf_img = cropped_img
                unknown_score = extract_hauf_safeguard(model, cropped_hauf_img, device)
                
            # Stage 2: Feature extraction
            feature_vector = None
            if use_detector_retrieval:
                if best_idx != -1 and hasattr(pred_instances, 'features') and pred_instances.features is not None:
                    feature_vector = pred_instances.features[best_idx].cpu().numpy()
                
                # Fallback if no box detected or features not populated
                if feature_vector is None:
                    data_sample = DetDataSample()
                    data_sample.set_metainfo({
                        'img_shape': img_bgr.shape[:2],
                        'ori_shape': img_bgr.shape[:2],
                        'pad_shape': img_bgr.shape[:2],
                    })
                    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        batch_inputs, batch_data_samples = model.data_preprocessor(img_tensor, [data_sample])
                        img_feats, txt_feats = model.extract_feat(batch_inputs, batch_data_samples)
                        raw_outs = model.bbox_head(img_feats, txt_feats)
                        ret_embeds = raw_outs[2] # list of tensors of shape (1, retrieval_dim, h_i, w_i)
                        
                        pooled_levels = []
                        for feat in ret_embeds:
                            pooled = feat.mean(dim=(2, 3)) # shape (1, retrieval_dim)
                            pooled_levels.append(pooled)
                        feature_vector = torch.stack(pooled_levels, dim=0).mean(dim=0)[0]
                        feature_vector = F.normalize(feature_vector, p=2, dim=0).cpu().numpy()
            
            processed_records.append({
                "image_path": item['image_path'],
                "class_id": item['class_id'],
                "class_label": item['class_label'],
                "feature_vector": feature_vector,
                "unknown_score": float(unknown_score)
            })
            
            if not use_detector_retrieval:
                crops.append(cropped_img)
                valid_indices.append(len(processed_records) - 1)
                
        except Exception as e:
            continue
            
    # Bước 2: Trích xuất song song theo Batch bằng CLIP
    if not use_detector_retrieval and crops:
        num_crops = len(crops)
        for i in tqdm(range(0, num_crops, batch_size), desc=f"{desc} (CLIP Batch Inference)"):
            batch_crops = crops[i:i + batch_size]
            batch_idxs = valid_indices[i:i + batch_size]
            
            try:
                inputs = clip_processor(images=batch_crops, return_tensors="pt", padding=True).to(device)
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
                    features_np = features.cpu().numpy()
                    
                for j, f_np in enumerate(features_np):
                    processed_records[batch_idxs[j]]["feature_vector"] = f_np
            except Exception as e:
                print(f"Error extracting batch {i}: {e}")
                
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
    gallery_split: str,
    auroc: float = None,
    fpr95: float = None,
    config_name: str = None
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
        
        if auroc is not None and fpr95 is not None:
            f.write("## Open-World Anomaly Detection Metrics (HAUF Safeguard)\n\n")
            f.write("| Metric | Score |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **AUROC (Area Under ROC)** | {auroc:.4f} |\n")
            f.write(f"| **FPR@TPR95** | {fpr95:.4f} |\n\n")
            
            if config_name:
                f.write(f"### ROC Curve Plot\n\n")
                f.write(f"![ROC Curve](roc_curve_{config_name}.png)\n\n")
        
        f.write("## Class-Wise Retrieval Metrics\n\n")
        f.write("| Class Name | Query Count | Recall@1 | Recall@5 | Recall@10 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls_name, metrics in sorted(class_metrics.items()):
            f.write(f"| {cls_name} | {metrics['count']} | {metrics['R@1']:.4f} | {metrics['R@5']:.4f} | {metrics['R@10']:.4f} |\n")
            
    print(f"-> Saved markdown evaluation report to: {report_path}")


def main():
    args = parse_args()
    
    # Convert local Kaggle CLIP model to safetensors if needed to bypass PyTorch < 2.6 CVE-2025-32434 check
    local_clip_path = "/kaggle/input/models/yujkaggle/openaiclip-vit-base-patch32/pytorch/default/1"
    working_clip_path = "/kaggle/working/openaiclip-vit-base-patch32"
    if args.clip_model == local_clip_path or (os.path.exists(local_clip_path) and os.path.samefile(args.clip_model, local_clip_path) if os.path.exists(args.clip_model) else False):
        if os.path.exists(local_clip_path) and not os.path.exists(os.path.join(working_clip_path, "model.safetensors")):
            print("-> Converting local pytorch_model.bin to safetensors to bypass PyTorch < 2.6 security restriction...")
            import shutil
            from safetensors.torch import save_file
            os.makedirs(working_clip_path, exist_ok=True)
            for fname in os.listdir(local_clip_path):
                if fname != "pytorch_model.bin":
                    shutil.copy(os.path.join(local_clip_path, fname), os.path.join(working_clip_path, fname))
            state_dict = torch.load(os.path.join(local_clip_path, "pytorch_model.bin"), map_location="cpu")
            state_dict = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
            save_file(state_dict, os.path.join(working_clip_path, "model.safetensors"))
            print(f"-> Successfully converted and saved safetensors to: {working_clip_path}")
        if os.path.exists(working_clip_path):
            args.clip_model = working_clip_path

    print("="*60)
    print("      IP102 RETRIEVAL METRICS EVALUATION PIPELINE      ")
    print("="*60)
    
    # 1. Load Dataset Splits Info
    print("-> Reading annotations...")
    query_records = load_dataset_records(args.dataset_root, args.query_split)
    gallery_records = load_dataset_records(args.dataset_root, args.gallery_split)
    print(f"-> Found {len(query_records)} query images and {len(gallery_records)} gallery images.")

    # Check if we need to force re-extraction of queries because they lack 'unknown_score'
    force_reextract_query = False
    if os.path.exists(args.query_cache):
        try:
            with open(args.query_cache, "rb") as f:
                temp_cache = pickle.load(f)
                if len(temp_cache) > 0 and "unknown_score" not in temp_cache[0]:
                    print("-> Cached Query embeddings do not contain 'unknown_score'. Forcing re-extraction...")
                    force_reextract_query = True
        except Exception:
            force_reextract_query = True

    # 2. Extract Query features (or load cache)
    if os.path.exists(args.query_cache) and not force_reextract_query:
        print(f"-> Loading pre-computed Query embeddings from cache: {args.query_cache}")
        with open(args.query_cache, "rb") as f:
            query_processed = pickle.load(f)
    else:
        print("-> Query cache not found or incomplete. Extracting embeddings on the fly...")
        model = init_detector(args.config, args.checkpoint, device=args.device)
        model.eval()
        
        if args.detector_retrieval:
            clip_model = None
            clip_processor = None
            print("-> Using detector's own retrieval head for feature extraction (no CLIP).")
        else:
            clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True, low_cpu_mem_usage=False).to(args.device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
            clip_model.eval()
        
        query_processed = extract_split_embeddings(
            query_records, model, clip_model, clip_processor, args.device, args.score_thr, "Processing Query Split",
            use_detector_retrieval=args.detector_retrieval
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
            model.eval()
            
        if args.detector_retrieval:
            clip_model = None
            clip_processor = None
        else:
            if 'clip_model' not in locals() or clip_model is None:
                use_safe = True
                if os.path.isdir(args.clip_model):
                    has_safe = any(f.endswith('.safetensors') for f in os.listdir(args.clip_model))
                    if not has_safe:
                        use_safe = False
                clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=use_safe, low_cpu_mem_usage=False).to(args.device)
                clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
                clip_model.eval()
            
        gallery_processed = extract_split_embeddings(
            gallery_records, model, clip_model, clip_processor, args.device, args.score_thr, "Processing Gallery Split",
            use_detector_retrieval=args.detector_retrieval
        )
        # Ensure we add unknown_score field to gallery too, for format consistency
        for item in gallery_processed:
            if "unknown_score" not in item:
                item["unknown_score"] = 0.0
        with open(args.gallery_cache, "wb") as f:
            pickle.dump(gallery_processed, f)
        print(f"-> Saved Gallery cache to {args.gallery_cache}")

    # 4. Perform Retrieval Evaluation
    print("-> Calculating retrieval metrics...")
    metrics, r1, r5, r10 = evaluate_retrieval(query_processed, gallery_processed)
    
    # 5. Perform Open-World Anomaly Detection Evaluation (AUROC & FPR@95)
    print("-> Calculating Open-World anomaly detection metrics...")
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    prev_intro_cls = cfg.get('prev_intro_cls', 0)
    cur_intro_cls = cfg.get('cur_intro_cls', 27)
    if 'model' in cfg and 'bbox_head' in cfg.model:
        prev_intro_cls = cfg.model.bbox_head.get('prev_intro_cls', prev_intro_cls)
        cur_intro_cls = cfg.model.bbox_head.get('cur_intro_cls', cur_intro_cls)
    num_known_classes = prev_intro_cls + cur_intro_cls
    print(f"-> Task class configuration: prev_intro_cls={prev_intro_cls}, cur_intro_cls={cur_intro_cls} (Total Known: {num_known_classes})")
    
    # Load category mapping from train.json to map class_id to 0-indexed position
    # In IP102 dataset, category IDs in the COCO annotations (e.g. 14, 15, 101) correspond 
    # to 1-based class IDs from classes.txt. Since the model expects 0-indexed indices (0 to 101),
    # we map each category ID x to x - 1.
    train_json_path = os.path.join(args.dataset_root, "train.json")
    cat_id_to_idx = {}
    if os.path.exists(train_json_path):
        try:
            with open(train_json_path, "r", encoding="utf-8") as f:
                coco_train = json.load(f)
            cat_id_to_idx = {cat['id']: cat['id'] - 1 for cat in coco_train.get('categories', [])}
        except Exception as e:
            print(f"Warning: Failed to parse train.json for category indices: {e}")
            
    if not cat_id_to_idx:
        all_class_ids = sorted(list(set([r['class_id'] for r in query_processed] + [r['class_id'] for r in gallery_processed])))
        cat_id_to_idx = {cid: cid - 1 for cid in all_class_ids}
        
    y_true = []
    y_scores = []
    for item in query_processed:
        class_idx = cat_id_to_idx.get(item['class_id'], 999)
        # 0 for Known, 1 for Unknown (unseen)
        label = 0 if class_idx < num_known_classes else 1
        y_true.append(label)
        y_scores.append(item.get('unknown_score', 0.0))
        
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    num_pos = np.sum(y_true)
    num_neg = len(y_true) - num_pos
    print(f"-> Query set OOD breakdown: {num_neg} Known samples, {num_pos} Unknown (unseen) samples.")
    
    auroc = None
    fpr95 = None
    config_name = Path(args.config).stem
    
    if num_pos > 0 and num_neg > 0:
        auroc, fpr, tpr = calculate_auroc_numpy(y_true, y_scores)
        fpr95 = calculate_fpr_at_tpr95(fpr, tpr)
        print(f"-> Open-World AUROC:   {auroc:.4f}")
        print(f"-> FPR@TPR95:          {fpr95:.4f}")
        
        # Plot ROC curve
        import matplotlib.pyplot as plt
        output_dir = os.path.dirname(args.output_report) if os.path.dirname(args.output_report) else "."
        plot_path = os.path.join(output_dir, f"roc_curve_{config_name}.png")
        try:
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auroc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter([fpr95], [0.95], color='red', zorder=5, label=f'FPR@TPR95 = {fpr95:.4f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(f'ROC Curve for Unknown Pest Detection\n({config_name})')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"-> Saved ROC curve plot to: {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to plot ROC curve: {e}")
    else:
        print("-> Skipped AUROC calculation because either Known or Unknown query samples are missing.")
    
    # 6. Display Summary
    print("\n" + "="*50)
    print("            SUMMARY RETRIEVAL METRICS            ")
    print("="*50)
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    if auroc is not None:
        print(f"AUROC:     {auroc:.4f}")
        print(f"FPR@TPR95: {fpr95:.4f}")
    print("="*50)
    
    # 7. Save Markdown Report
    save_report(
        args.output_report, 
        metrics, 
        r1, 
        r5, 
        r10, 
        args.query_split, 
        args.gallery_split,
        auroc=auroc,
        fpr95=fpr95,
        config_name=config_name
    )
    print("====== Evaluation Process Completed Successfully! ======")


if __name__ == "__main__":
    main()
