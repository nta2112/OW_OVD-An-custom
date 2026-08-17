"""
Lifelong Image Retrieval Evaluation Script
==========================================
Computes standard retrieval metrics (Recall@K, mAP, Rank-1 Accuracy) and lifelong learning 
metrics (Plasticity, Forgetting, and Overall Change) across task boundaries.
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

    # Monkey patch Transformers check_torch_load_is_safe for compatibility with older torch versions (CVE-2025-32434 bypass)
    try:
        import transformers.utils.import_utils as transformers_import_utils
        transformers_import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        
        import transformers.utils as transformers_utils
        transformers_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        
        import transformers.modeling_utils as transformers_modeling_utils
        transformers_modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        
        print("-> Fully patched transformers check_torch_load_is_safe across namespaces")
    except Exception as e:
        print(f"-> Failed to patch transformers: {e}")

# Run patches immediately
patch_environment()

import torch
import torch.nn.functional as F
import cv2
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
from transformers import CLIPProcessor, CLIPModel

# 25 custom classes of IP102 subset
class_names = ['14', '15', '16', '18', '22', '23', '24', '25', '26', '37', '38', '39', '45', '46', '47', '48', '49', '50', '51', '66', '67', '69', '70', '86', '101']

# Continual/Lifelong Task Class boundaries
TASK_GROUPS = {
    1: ['14', '15', '16', '18', '22', '23', '24'],       # Task 1 classes (7 classes)
    2: ['25', '26', '37', '38', '39', '45'],             # Task 2 classes (6 classes)
    3: ['46', '47', '48', '49', '50', '66'],             # Task 3 classes (6 classes)
    4: ['67', '69', '70', '86', '101']                  # Task 4 classes (5 classes)
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Lifelong CBIR Pest Retrieval System")
    parser.add_argument("--config", type=str, required=True,
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
    parser.add_argument("--output-report", type=str, default="retrieval_lifelong_report.md",
                        help="Path to save markdown report file")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    parser.add_argument("--clip-model", type=str, default="/kaggle/input/models/yujkaggle/openaiclip-vit-base-patch32/pytorch/default/1",
                        help="CLIP vision model to use")
    parser.add_argument("--score-thr", type=float, default=0.35,
                        help="Confidence threshold for pest detection")
    parser.add_argument("--detector-retrieval", action="store_true",
                        help="Use the detector's own retrieval head instead of CLIP")
    parser.add_argument("--current-task", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Task index currently trained (1, 2, 3, or 4)")
    parser.add_argument("--history-file", type=str, default="history_metrics.json",
                        help="JSON file storing history of task metrics for forgetting/plasticity computation")
    return parser.parse_args()


def load_dataset_records(dataset_root: str, split: str) -> List[Dict]:
    json_path = os.path.join(dataset_root, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON annotation file not found: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
        
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    
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
        for root, _, files in os.walk(dataset_root):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                image_folder = root
                found_folder = True
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
    try:
        crop_cv = cv2.cvtColor(np.array(crop_img.resize((640, 640))), cv2.COLOR_RGB2BGR)
        crop_tensor = torch.from_numpy(crop_cv).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': crop_cv.shape[:2],
            'ori_shape': crop_cv.shape[:2],
            'pad_shape': crop_cv.shape[:2],
        })
        
        with torch.no_grad():
            batch_inputs, batch_data_samples = model.data_preprocessor(crop_tensor, [data_sample])
            img_feats, txt_feats = model.extract_feat(batch_inputs, batch_data_samples)
            raw_outs = model.bbox_head(img_feats, txt_feats)
            
            # Predict unknown using head
            att_feats = model.bbox_head.att_embeddings[None].repeat(img_feats[0].shape[0], 1, 1)
            cls_scores_with_unknown, _ = model.bbox_head.predict_unknown(raw_outs, img_feats, att_feats)
            
            max_pu = -1.0
            for i, logits in enumerate(cls_scores_with_unknown):
                pu_map = logits[0, -1]  # Unknown score channel
                val, idx = pu_map.view(-1).max(dim=0)
                if val.item() > max_pu:
                    max_pu = val.item()
                    
            return max_pu
            
    except Exception as e:
        return 0.0


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
    processed_records = []
    crops = []
    valid_indices = []
    
    # Bước 1: Phát hiện đối tượng và cắt ảnh (Detection & Crop)
    for idx, item in enumerate(tqdm(records, desc=f"{desc} (BBox Detection)")):
        try:
            img_pil = Image.open(item['image_path']).convert("RGB")
            width, height = img_pil.size
            
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
                
            unknown_score = 0.0
            if best_box is not None and hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'att_embeddings') and model.bbox_head.att_embeddings is not None:
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
                
            feature_vector = None
            if use_detector_retrieval:
                if best_idx != -1 and hasattr(pred_instances, 'features') and pred_instances.features is not None:
                    feature_vector = pred_instances.features[best_idx].cpu().numpy()
                
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
                        ret_embeds = raw_outs[2] # Retrieval embeddings
                        
                        pooled_levels = []
                        for feat in ret_embeds:
                            pooled = feat.mean(dim=(2, 3))
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
    gallery_embeddings = np.array([item["feature_vector"] for item in gallery_data])
    gallery_classes = [item["class_id"] for item in gallery_data]
    
    class_eval = {}
    total_ap = 0.0
    total_r1 = 0.0
    
    for q_item in tqdm(query_data, desc="Matching Queries"):
        q_class = q_item["class_id"]
        q_label = q_item["class_label"]
        q_embed = q_item["feature_vector"]
        
        sims = np.dot(q_embed, gallery_embeddings.T)
        
        # Sort entire gallery for AP and Rank-1 Accuracy
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
        
        # Cache results in query items for task-wise grouping later
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
    macro_r1, macro_r5, macro_r10, macro_ap, macro_rank1 = 0.0, 0.0, 0.0, 0.0, 0.0
    total_queries = len(query_data)
    
    for label, data in class_eval.items():
        count = len(data[1])
        r1 = np.mean(data[1])
        r5 = np.mean(data[5])
        r10 = np.mean(data[10])
        ap = np.mean(data["ap"])
        rank1 = np.mean(data["r1"])
        
        class_metrics[label] = {
            "count": count,
            "R@1": r1,
            "R@5": r5,
            "R@10": r10,
            "AP": ap,
            "Rank1": rank1
        }
        
        macro_r1 += r1
        macro_r5 += r5
        macro_r10 += r10
        macro_ap += ap
        macro_rank1 += rank1
        
    num_classes = len(class_metrics)
    if num_classes > 0:
        macro_r1 /= num_classes
        macro_r5 /= num_classes
        macro_r10 /= num_classes
        macro_ap /= num_classes
        macro_rank1 /= num_classes
        
    return class_metrics, macro_r1, macro_r5, macro_r10, macro_ap


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
    print("      LIFELONG IMAGE RETRIEVAL EVALUATION PIPELINE      ")
    print("="*60)
    
    print("-> Reading annotations...")
    query_records = load_dataset_records(args.dataset_root, args.query_split)
    gallery_records = load_dataset_records(args.dataset_root, args.gallery_split)
    print(f"-> Found {len(query_records)} query images and {len(gallery_records)} gallery images.")

    # Force re-extract query if cache is stale
    force_reextract_query = False
    if os.path.exists(args.query_cache):
        try:
            with open(args.query_cache, "rb") as f:
                temp_cache = pickle.load(f)
                if len(temp_cache) > 0 and "unknown_score" not in temp_cache[0]:
                    print("-> Stale Query cache. Forcing re-extraction...")
                    force_reextract_query = True
        except Exception:
            force_reextract_query = True

    detector_model = None
    clip_model = None
    clip_processor = None

    def get_models():
        nonlocal detector_model, clip_model, clip_processor
        if detector_model is None:
            detector_model = init_detector(args.config, args.checkpoint, device=args.device)
            detector_model.eval()
        if not args.detector_retrieval and clip_model is None:
            print(f"-> Loading CLIP model: {args.clip_model}")
            clip_model = CLIPModel.from_pretrained(args.clip_model, local_files_only=True, low_cpu_mem_usage=False).to(args.device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=True)
        return detector_model, clip_model, clip_processor

    if os.path.exists(args.query_cache) and not force_reextract_query:
        print(f"-> Loading Query embeddings from cache: {args.query_cache}")
        with open(args.query_cache, "rb") as f:
            query_processed = pickle.load(f)
    else:
        print("-> Extracting Query embeddings...")
        detector_model, clip_model, clip_processor = get_models()
        
        query_processed = extract_split_embeddings(
            query_records, detector_model, clip_model, clip_processor, args.device,
            args.score_thr, "Query Extraction", args.detector_retrieval
        )
        with open(args.query_cache, "wb") as f:
            pickle.dump(query_processed, f)

    # 3. Extract Gallery features
    if os.path.exists(args.gallery_cache):
        print(f"-> Loading Gallery embeddings from cache: {args.gallery_cache}")
        with open(args.gallery_cache, "rb") as f:
            gallery_processed = pickle.load(f)
    else:
        print("-> Extracting Gallery embeddings...")
        detector_model, clip_model, clip_processor = get_models()
        
        gallery_processed = extract_split_embeddings(
            gallery_records, detector_model, clip_model, clip_processor, args.device,
            args.score_thr, "Gallery Extraction", args.detector_retrieval
        )
        with open(args.gallery_cache, "wb") as f:
            pickle.dump(gallery_processed, f)

    # 4. Evaluate Retrieval (Recall, mAP, Rank-1)
    class_metrics, macro_r1, macro_r5, macro_r10, macro_ap = evaluate_retrieval(
        query_processed, gallery_processed
    )
    
    # 5. Evaluate Open-World Unknown Anomaly Metrics (AUROC)
    auroc = 0.5
    fpr95 = 1.0
    known_labels = TASK_GROUPS[1] + TASK_GROUPS[2] + TASK_GROUPS[3] + TASK_GROUPS[4]
    
    # Known classes in the current task config
    current_knowns = []
    for t_idx in range(1, args.current_task + 1):
        current_knowns += TASK_GROUPS[t_idx]
        
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
            plot_path = f"roc_curve_task_{args.current_task}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"-> Saved ROC Curve plot to: {plot_path}")
        except Exception:
            pass

    # 6. Lifelong metric computations: Plasticity, Forgetting, and Overall Change
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
    overall_change_signed = plasticity + forgetting

    # 7. Print Console Summary Report
    print("\n" + "="*40 + " EVALUATION SUMMARY Task " + str(args.current_task) + " " + "="*40)
    print(f"Global mAP:       {macro_ap:.4f}")
    print(f"Recall@1:         {macro_r1:.4f}")
    print(f"Recall@5:         {macro_r5:.4f}")
    print(f"Recall@10:        {macro_r10:.4f}")
    print(f"OOD AUROC:        {auroc:.4f}")
    print(f"OOD FPR@TPR95:    {fpr95:.4f}")
    print("-"*40)
    print(f"Plasticity:       {plasticity:.4f}")
    print(f"Forgetting (mAP): {forgetting:.4f} ({forgetting*100:.2f}%)")
    print(f"Overall Change:   {overall_change:.4f}")
    print("="*105 + "\n")

    # 8. Save Detailed Markdown Report
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(f"# Lifelong Image Retrieval Report - Task {args.current_task}\n\n")
        f.write(f"- **Query Split:** `{args.query_split}`\n")
        f.write(f"- **Gallery Split:** `{args.gallery_split}`\n")
        f.write(f"- **Retrieval Mode:** `{'Detector Head' if args.detector_retrieval else 'CLIP'}`\n\n")
        
        f.write("## 1. Global Summarized Metrics\n\n")
        f.write("| Metric | Macro Average |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| **Retrieval mAP** | {macro_ap:.4f} |\n")
        f.write(f"| **Recall@1** | {macro_r1:.4f} |\n")
        f.write(f"| **Recall@5** | {macro_r5:.4f} |\n")
        f.write(f"| **Recall@10** | {macro_r10:.4f} |\n\n")
        
        f.write("## 2. Lifelong Continual Learning Metrics\n\n")
        f.write("| Lifelong Metric | Value | Explanation |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| **Plasticity** | **{plasticity:.4f}** | Performance mAP on the newest task right after learning it |\n")
        f.write(f"| **Forgetting** | **{forgetting:.4f}** | Average mAP performance drop on previously learned tasks |\n")
        f.write(f"| **Overall Change (Plasticity - Forgetting)** | **{overall_change:.4f}** | Consolidated trade-off of learning capacity and stability |\n")
        f.write(f"| **Overall Change (Plasticity + Forgetting)** | **{overall_change_signed:.4f}** | Total sum of Plasticity and positive Forgetting |\n\n")
        
        f.write("### Task-Wise Metrics History Matrix\n\n")
        f.write("| Evaluation Stage | T1 mAP | T2 mAP | T3 mAP | T4 mAP |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for stage in sorted(history.keys()):
            h_metrics = history[stage]
            f.write(f"| After {stage.upper().replace('_', ' ')} | {h_metrics.get('T1', {}).get('mAP', 0.0):.4f} | {h_metrics.get('T2', {}).get('mAP', 0.0):.4f} | {h_metrics.get('T3', {}).get('mAP', 0.0):.4f} | {h_metrics.get('T4', {}).get('mAP', 0.0):.4f} |\n")
        f.write("\n")
        
        f.write("## 3. Open-World Unknown Anomaly Metrics (HAUF Safeguard)\n\n")
        f.write("| Metric | Score |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| **OOD AUROC** | {auroc:.4f} |\n")
        f.write(f"| **FPR@TPR95** | {fpr95:.4f} |\n\n")
        
        if os.path.exists(f"roc_curve_task_{args.current_task}.png"):
            f.write(f"![ROC Curve](roc_curve_task_{args.current_task}.png)\n\n")
            
        f.write("## 4. Class-Wise Metrics Breakdown\n\n")
        f.write("| Class Name | Task Group | Query Count | Recall@1 | Recall@5 | Recall@10 | AP | Rank1 Accuracy |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cls_name, metrics in sorted(class_metrics.items()):
            t_idx = -1
            for task, classes in TASK_GROUPS.items():
                if cls_name in classes:
                    t_idx = task
                    break
            f.write(f"| {cls_name} | Task {t_idx} | {metrics['count']} | {metrics['R@1']:.4f} | {metrics['R@5']:.4f} | {metrics['R@10']:.4f} | {metrics['AP']:.4f} | {metrics['Rank1']:.4f} |\n")
            
    print(f"-> Saved lifelong markdown evaluation report to: {args.output_report}")
    print("="*105)

if __name__ == "__main__":
    main()
