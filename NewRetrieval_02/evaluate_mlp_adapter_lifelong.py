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

# Add parent directory of this script to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

    # Patch version limits BEFORE importing mmdet/mmyolo to prevent AssertionError
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

    # Spoof mmcv version to bypass strict MMCV maximum version checks in mmdet/mmyolo
    try:
        import mmcv
        mmcv.__version__ = '2.0.1'
    except Exception:
        pass

    # Monkey patch Config._file2dict for relative path configs
    try:
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
    except Exception:
        pass

    # Monkey patch Transformers check_torch_load_is_safe for compatibility
    try:
        import transformers.utils.import_utils as transformers_import_utils
        transformers_import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        
        import transformers.utils as transformers_utils
        transformers_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        
        import transformers.modeling_utils as transformers_modeling_utils
        transformers_modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    except Exception:
        pass

# Run patches immediately
patch_environment()

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
from image_retrieval import load_feature_extractor, extract_visual_features
from train_mlp_adapter_lifelong import MLPAdapter, TASK_GROUPS

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Lifelong CBIR Pest Retrieval System using MLP Adapter")
    parser.add_argument("--config", type=str, required=True, help="Path to detector config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to detector checkpoint file")
    parser.add_argument("--mlp-checkpoint", type=str, required=True, help="Path to the trained MLP adapter checkpoint file")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to IP102 dataset root directory")
    parser.add_argument("--query-split", type=str, default="val", help="Dataset split to use as Query (e.g. val, test)")
    parser.add_argument("--gallery-split", type=str, default="test", help="Dataset split to use as Gallery (e.g. test, train)")
    parser.add_argument("--query-cache", type=str, default="query_cache_mlp.pkl", help="Path to save/load query base embeddings cache")
    parser.add_argument("--gallery-cache", type=str, default="gallery_cache_mlp.pkl", help="Path to save/load gallery base embeddings cache")
    parser.add_argument("--output-report", type=str, default="retrieval_lifelong_report_mlp.md", help="Path to save markdown report file")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--extractor-type", type=str, default="vit", choices=["clip", "dinov2", "vit"], help="Feature extractor type")
    parser.add_argument("--extractor-model", type=str, required=True, help="Feature extractor model path/name")
    parser.add_argument("--score-thr", type=float, default=0.35, help="Confidence threshold for pest detection")
    parser.add_argument("--current-task", type=int, default=1, choices=[1, 2, 3, 4], help="Current trained task index")
    parser.add_argument("--history-file", type=str, default="history_metrics_mlp.json", help="JSON file storing history of task metrics")
    return parser.parse_args()

def load_dataset_records(dataset_root: str, split: str) -> List[Dict]:
    json_path = os.path.join(dataset_root, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON annotation file not found: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
        
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    
    # Index all image paths in dataset_root recursively once to avoid recursive glob inside the loop
    print(f"-> Scanning dataset root to index file paths for split '{split}'...")
    image_path_map = {}
    for root, _, files in os.walk(dataset_root):
        for f_name in files:
            if f_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path_map[f_name] = os.path.join(root, f_name)
    print(f"-> Indexed {len(image_path_map)} images.")
    
    images = coco.get('images', [])
    img_id_to_info = {}
    for img in images:
        file_name = os.path.basename(img['file_name'])
        img_id_to_info[img['id']] = {
            'file_name': file_name,
            'image_path': image_path_map.get(file_name)
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
        if real_path is None or not os.path.exists(real_path):
            continue
                
        records.append({
            "image_path": real_path,
            "class_id": class_id,
            "class_label": cat_id_to_name.get(class_id, f"class_{class_id}")
        })
        
    return records

def extract_split_base_embeddings(
    records: List[Dict], 
    detector_model, 
    extractor_model, 
    extractor_processor, 
    device: str, 
    score_thr: float,
    desc: str,
    extractor_type: str = "vit",
    batch_size: int = 64
) -> List[Dict]:
    processed_records = []
    crops = []
    valid_indices = []
    
    # Step 1: Detect and crop
    for idx, item in enumerate(tqdm(records, desc=f"{desc} (BBox Detection)")):
        try:
            img_pil = Image.open(item['image_path']).convert("RGB")
            width, height = img_pil.size
            
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            det_result = inference_detector(detector_model, img_bgr)
            pred_instances = det_result.pred_instances
            boxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            
            best_box = None
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                if scores[best_idx] >= score_thr:
                    best_box = boxes[best_idx]
                    
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
                
            processed_records.append({
                "image_path": item['image_path'],
                "class_id": item['class_id'],
                "class_label": item['class_label'],
                "base_feature": None
            })
            
            crops.append(cropped_img)
            valid_indices.append(len(processed_records) - 1)
            
        except Exception:
            continue
            
    # Step 2: Batch extract frozen features
    if crops:
        num_crops = len(crops)
        for i in tqdm(range(0, num_crops, batch_size), desc=f"{desc} ({extractor_type.upper()} Batch Inference)"):
            batch_crops = crops[i:i + batch_size]
            batch_idxs = valid_indices[i:i + batch_size]
            
            try:
                with torch.no_grad():
                    features = extract_visual_features(
                        model=extractor_model,
                        processor=extractor_processor,
                        images=batch_crops,
                        extractor_type=extractor_type,
                        device=device
                    )
                    # Convert to CPU numpy array (already L2 normalized by extract_visual_features)
                    features_np = features.cpu().numpy()
                    
                for j, f_np in enumerate(features_np):
                    processed_records[batch_idxs[j]]["base_feature"] = f_np
            except Exception as e:
                print(f"Error extracting batch {i}: {e}")
                
    return processed_records

def compute_ap(q_class: int, ranked_classes: List[int]) -> float:
    ap = 0.0
    hits = 0
    total_pos = sum(1 for g_class in ranked_classes if g_class == q_class)
    if total_pos == 0:
        return 0.0
    for rank, g_class in enumerate(ranked_classes):
        if g_class == q_class:
            hits += 1
            precision = hits / (rank + 1)
            ap += precision
            if hits == total_pos:
                break
    return ap / total_pos

def evaluate_retrieval(query_data: List[Dict], gallery_data: List[Dict]) -> Tuple[Dict, float, float, float, float]:
    gallery_embeddings = np.array([item["retrieval_embedding"] for item in gallery_data])
    gallery_classes = [item["class_id"] for item in gallery_data]
    
    class_eval = {}
    total_ap = 0.0
    total_r1 = 0.0
    
    for q_item in tqdm(query_data, desc="Matching Queries"):
        q_class = q_item["class_id"]
        q_label = q_item["class_label"]
        q_embed = q_item["retrieval_embedding"]
        
        sims = np.dot(q_embed, gallery_embeddings.T)
        
        # Sort indices descending
        full_rank_indices = np.argsort(sims)[::-1]
        full_ranked_classes = [gallery_classes[idx] for idx in full_rank_indices]
        
        top_classes = full_ranked_classes[:10]
        
        if q_label not in class_eval:
            class_eval[q_label] = {1: [], 5: [], 10: [], "ap": [], "r1": []}
            
        rec1 = 1.0 if q_class in top_classes[:1] else 0.0
        rec5 = 1.0 if q_class in top_classes[:5] else 0.0
        rec10 = 1.0 if q_class in top_classes[:10] else 0.0
        
        ap = compute_ap(q_class, full_ranked_classes)
        r1 = rec1
        
        q_item["ap"] = ap
        q_item["r1"] = r1
        
        class_eval[q_label][1].append(rec1)
        class_eval[q_label][5].append(rec5)
        class_eval[q_label][10].append(rec10)
        class_eval[q_label]["ap"].append(ap)
        class_eval[q_label]["r1"].append(r1)
        
        total_ap += ap
        total_r1 += r1
        
    class_metrics = {}
    macro_r1, macro_r5, macro_r10, macro_ap = 0.0, 0.0, 0.0, 0.0
    
    for label, data in class_eval.items():
        count = len(data[1])
        r1 = np.mean(data[1])
        r5 = np.mean(data[5])
        r10 = np.mean(data[10])
        ap = np.mean(data["ap"])
        
        class_metrics[label] = {
            "count": count,
            "R@1": r1,
            "R@5": r5,
            "R@10": r10,
            "AP": ap
        }
        
        macro_r1 += r1
        macro_r5 += r5
        macro_r10 += r10
        macro_ap += ap
        
    num_classes = len(class_metrics)
    if num_classes > 0:
        macro_r1 /= num_classes
        macro_r5 /= num_classes
        macro_r10 /= num_classes
        macro_ap /= num_classes
        
    return class_metrics, macro_r1, macro_r5, macro_r10, macro_ap

def calculate_auroc_numpy(y_true, y_scores) -> Tuple[float, np.ndarray, np.ndarray]:
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
    
    area = 0.0
    for i in range(len(fpr) - 1):
        area += 0.5 * (tpr[i] + tpr[i+1]) * (fpr[i+1] - fpr[i])
        
    return area, fpr, tpr

def calculate_fpr_at_tpr95(fpr, tpr) -> float:
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    return 1.0

def main():
    args = parse_args()
    
    print("="*60)
    print("      LIFELONG CBIR RETRIEVAL EVALUATION (MLP ADAPTER)      ")
    print("="*60)
    
    # 1. Load data records
    query_records = load_dataset_records(args.dataset_root, args.query_split)
    gallery_records = load_dataset_records(args.dataset_root, args.gallery_split)
    print(f"-> Found {len(query_records)} query images and {len(gallery_records)} gallery images.")
    
    detector_model = None
    extractor_model = None
    extractor_processor = None
    
    def get_models():
        nonlocal detector_model, extractor_model, extractor_processor
        if detector_model is None:
            detector_model = init_detector(args.config, args.checkpoint, device=args.device)
            detector_model.eval()
        if extractor_model is None:
            extractor_model, extractor_processor = load_feature_extractor(
                model_name=args.extractor_model,
                extractor_type=args.extractor_type,
                device=args.device
            )
        return detector_model, extractor_model, extractor_processor

    # 2. Extract Base Features (using cache if available)
    if os.path.exists(args.query_cache):
        print(f"-> Loading Query base embeddings from: {args.query_cache}")
        with open(args.query_cache, "rb") as f:
            query_processed = pickle.load(f)
    else:
        print("-> Extracting Query base embeddings...")
        detector_model, extractor_model, extractor_processor = get_models()
        query_processed = extract_split_base_embeddings(
            query_records, detector_model, extractor_model, extractor_processor, args.device,
            args.score_thr, "Query Extraction", args.extractor_type
        )
        with open(args.query_cache, "wb") as f:
            pickle.dump(query_processed, f)
            
    if os.path.exists(args.gallery_cache):
        print(f"-> Loading Gallery base embeddings from: {args.gallery_cache}")
        with open(args.gallery_cache, "rb") as f:
            gallery_processed = pickle.load(f)
    else:
        print("-> Extracting Gallery base embeddings...")
        detector_model, extractor_model, extractor_processor = get_models()
        gallery_processed = extract_split_base_embeddings(
            gallery_records, detector_model, extractor_model, extractor_processor, args.device,
            args.score_thr, "Gallery Extraction", args.extractor_type
        )
        with open(args.gallery_cache, "wb") as f:
            pickle.dump(gallery_processed, f)
            
    # 3. Load MLP Adapter and project features
    print(f"-> Loading trained MLP adapter from: {args.mlp_checkpoint}")
    input_dim = query_processed[0]["base_feature"].shape[0]
    mlp = MLPAdapter(input_dim=input_dim, output_dim=256).to(args.device)
    mlp.load_state_dict(torch.load(args.mlp_checkpoint, map_location=args.device))
    mlp.eval()
    
    # Project features
    with torch.no_grad():
        for q_item in query_processed:
            feat_tensor = torch.tensor(q_item["base_feature"], dtype=torch.float32).unsqueeze(0).to(args.device)
            proj_feat = mlp(feat_tensor).squeeze(0).cpu().numpy()
            q_item["retrieval_embedding"] = proj_feat
            
        for g_item in gallery_processed:
            feat_tensor = torch.tensor(g_item["base_feature"], dtype=torch.float32).unsqueeze(0).to(args.device)
            proj_feat = mlp(feat_tensor).squeeze(0).cpu().numpy()
            g_item["retrieval_embedding"] = proj_feat
            
    # 4. Evaluate Retrieval
    class_metrics, macro_r1, macro_r5, macro_r10, macro_ap = evaluate_retrieval(
        query_processed, gallery_processed
    )
    
    # 5. Seen vs Unseen analysis & OOD Anomaly Detection
    known_labels = TASK_GROUPS[1] + TASK_GROUPS[2] + TASK_GROUPS[3] + TASK_GROUPS[4]
    current_knowns = []
    for t_idx in range(1, args.current_task + 1):
        current_knowns += TASK_GROUPS[t_idx]
    
    # Calculate OOD score (unknown_score) using Nearest Neighbor distance in retrieval space
    gallery_embeddings = np.array([item["retrieval_embedding"] for item in gallery_processed])
    gallery_labels = [item["class_label"] for item in gallery_processed]
    known_gallery_indices = [idx for idx, label in enumerate(gallery_labels) if label in current_knowns]
    
    if len(known_gallery_indices) > 0:
        for q_item in query_processed:
            q_embed = q_item["retrieval_embedding"]
            sims = np.dot(q_embed, gallery_embeddings[known_gallery_indices].T)
            max_known_sim = np.max(sims)
            q_item["unknown_score"] = float(1.0 - max_known_sim)
    else:
        for q_item in query_processed:
            q_item["unknown_score"] = 0.0

    seen_r1s = [q_item["r1"] for q_item in query_processed if q_item["class_label"] in current_knowns]
    unseen_r1s = [q_item["r1"] for q_item in query_processed if q_item["class_label"] not in current_knowns and q_item["class_label"] in known_labels]
    
    recall_seen_r1 = float(np.mean(seen_r1s)) if len(seen_r1s) > 0 else 0.0
    recall_unseen_r1 = float(np.mean(unseen_r1s)) if len(unseen_r1s) > 0 else None
    
    # Calculate AUROC and FPR@TPR95
    auroc = None
    fpr95 = None
    y_true_ood = []
    y_scores_ood = []
    for q_item in query_processed:
        q_label = q_item["class_label"]
        if q_label in known_labels:
            is_unknown = 0 if q_label in current_knowns else 1
            y_true_ood.append(is_unknown)
            y_scores_ood.append(q_item["unknown_score"])
            
    if len(y_true_ood) > 0 and sum(y_true_ood) > 0 and sum(y_true_ood) < len(y_true_ood):
        auroc, fpr, tpr = calculate_auroc_numpy(y_true_ood, y_scores_ood)
        fpr95 = calculate_fpr_at_tpr95(fpr, tpr)
        
        # Save ROC curve image
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:0.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (OOD)')
            plt.legend(loc="lower right")
            plot_path = f"roc_curve_task_{args.current_task}_mlp.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"-> Saved ROC Curve plot to: {plot_path}")
        except Exception:
            pass

    # 6. Lifelong metric computations (Plasticity, Forgetting, Overall Change)
    history = {}
    if os.path.exists(args.history_file):
        try:
            with open(args.history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            pass
            
    current_task_metrics = {}
    for t_idx in [1, 2, 3, 4]:
        group_classes = TASK_GROUPS[t_idx]
        group_aps = [q_item["ap"] for q_item in query_processed if q_item["class_label"] in group_classes]
        group_r1s = [q_item["r1"] for q_item in query_processed if q_item["class_label"] in group_classes]
        
        current_task_metrics[f"T{t_idx}"] = {
            "mAP": float(np.mean(group_aps)) if len(group_aps) > 0 else 0.0,
            "Rank1": float(np.mean(group_r1s)) if len(group_r1s) > 0 else 0.0
        }
        
    history[f"task_{args.current_task}"] = current_task_metrics
    with open(args.history_file, 'w') as f:
        json.dump(history, f, indent=4)
        
    plasticity = current_task_metrics[f"T{args.current_task}"]["mAP"]
    
    forgetting_list = []
    for i in range(1, args.current_task):
        peak_key = f"task_{i}"
        curr_key = f"task_{args.current_task}"
        if peak_key in history and curr_key in history:
            peak_val = history[peak_key][f"T{i}"]["mAP"]
            curr_val = history[curr_key][f"T{i}"]["mAP"]
            drop = peak_val - curr_val
            forgetting_list.append(max(0.0, drop))
            
    forgetting = float(np.mean(forgetting_list)) if len(forgetting_list) > 0 else 0.0
    overall_change = plasticity - forgetting
    
    # Helper formatter for print
    def fmt_val(v):
        return f"{v:.4f}" if v is not None else "None"

    # 7. Print summary report
    print("\n" + "="*40 + " EVALUATION SUMMARY (MLP ADAPTER) Task " + str(args.current_task) + " " + "="*40)
    print(f"Global mAP:       {macro_ap:.4f}")
    print(f"Recall@1:         {macro_r1:.4f}")
    print(f"Recall@5:         {macro_r5:.4f}")
    print(f"Recall@10:        {macro_r10:.4f}")
    print(f"Recall@1 (Seen):  {recall_seen_r1:.4f}")
    print(f"Recall@1 (Unseen):{fmt_val(recall_unseen_r1)}")
    print(f"OOD AUROC:        {fmt_val(auroc)}")
    print(f"OOD FPR@TPR95:    {fmt_val(fpr95)}")
    print("-"*40)
    print(f"Plasticity:       {plasticity:.4f}")
    print(f"Forgetting (mAP): {forgetting:.4f} ({forgetting*100:.2f}%)")
    print(f"Overall Change:   {overall_change:.4f}")
    print("="*105 + "\n")
    
    # 8. Save Detailed Markdown Report
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(f"# Lifelong MLP Adapter Retrieval Report (Task {args.current_task})\n\n")
        f.write(f"- **Extractor Type:** `{args.extractor_type}`\n")
        f.write(f"- **Base Extractor Model:** `{args.extractor_model}`\n")
        f.write(f"- **MLP Checkpoint:** `{args.mlp_checkpoint}`\n\n")
        f.write("## Lifelong Metrics\n\n")
        f.write("| Metric | Score | Description |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| **Plasticity** | **{plasticity:.4f}** | Performance mAP on newest task right after learning |\n")
        f.write(f"| **Forgetting** | **{forgetting:.4f}** | Average drop in performance on older tasks |\n")
        f.write(f"| **Overall Change** | **{overall_change:.4f}** | Trade-off of learning capacity and stability |\n\n")
        
        f.write("## Summary Retrieval Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| **Recall@1** | {macro_r1:.4f} |\n")
        f.write(f"| **Recall@5** | {macro_r5:.4f} |\n")
        f.write(f"| **Recall@10** | {macro_r10:.4f} |\n")
        f.write(f"| **Recall@1 (Seen)** | {recall_seen_r1:.4f} |\n")
        f.write(f"| **Recall@1 (Unseen)** | {f'{recall_unseen_r1:.4f}' if recall_unseen_r1 is not None else 'N/A'} |\n")
        f.write(f"| **OOD AUROC** | {f'{auroc:.4f}' if auroc is not None else 'N/A'} |\n")
        f.write(f"| **OOD FPR@TPR95** | {f'{fpr95:.4f}' if fpr95 is not None else 'N/A'} |\n\n")
        
    print(f"Saved evaluation report to: {args.output_report}")

if __name__ == "__main__":
    main()
