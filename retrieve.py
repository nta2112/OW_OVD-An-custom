"""
Online Query and Image Retrieval
================================
This script handles the search query pipeline. It loads a query image, detects the pest
using the OW-OVD model, crops the region, performs HAUF (Open-World Safeguard) score 
analysis, extracts the CLIP visual embedding, and retrieves the Top-K most similar
images from the gallery index.

Usage:
    python retrieve.py \
        --config configs/open_world/mowod/custom/ip102_t1.py \
        --checkpoint pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth \
        --gallery-index gallery_index.pkl \
        --query-image data/IP102/test/pest_1.jpg \
        --top-k 10 \
        --anomaly-thr 0.55 \
        --output retrieval_output.jpg \
        --device cuda:0
"""

import argparse
import os
import sys
import pickle
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path

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

    # Spoof mmcv version to bypass strict MMCV maximum version checks in mmdet/mmyolo
    try:
        import mmcv
        mmcv.__version__ = '2.0.1'
    except Exception:
        pass

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
from mmdet.structures import DetDataSample
from image_retrieval import load_feature_extractor, extract_visual_features


def parse_args():
    parser = argparse.ArgumentParser(description="Query and retrieve similar pest images")
    parser.add_argument("--config", type=str, default="configs/open_world/mowod/custom/ip102_t1.py",
                        help="Path to detector config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to detector checkpoint file")
    parser.add_argument("--gallery-index", type=str, default="gallery_index.pkl",
                        help="Path to gallery index database file")
    parser.add_argument("--query-image", type=str, required=True,
                        help="Path to query image file")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top retrieved images to return")
    parser.add_argument("--anomaly-thr", type=float, default=0.55,
                        help="Anomaly detection threshold for unknown pests")
    parser.add_argument("--output", type=str, default="retrieval_output.jpg",
                        help="Path to save the output visualization image")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    parser.add_argument("--extractor-type", type=str, default="clip", choices=["clip", "dinov2", "vit"],
                        help="Feature extractor type to use")
    parser.add_argument("--extractor-model", type=str, default=None,
                        help="Feature extractor model repository or local path")
    parser.add_argument("--clip-model", type=str, default=None,
                        help="Deprecated alias for --extractor-model")
    parser.add_argument("--score-thr", type=float, default=0.35,
                        help="Confidence threshold for pest detection")
    return parser.parse_args()


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
            best_level = 0
            best_y, best_x = 0, 0
            
            for i, logits in enumerate(cls_scores_with_unknown):
                pu_map = logits[0, -1]  # The last channel is P_u (already sigmoid-activated)
                val, idx = pu_map.view(-1).max(dim=0)
                if val.item() > max_pu:
                    max_pu = val.item()
                    best_level = i
                    best_y = idx.item() // pu_map.shape[1]
                    best_x = idx.item() % pu_map.shape[1]
                    
            # Compute HAUF details at the best cell
            known_logits_at_cell = cls_scores_with_unknown[best_level][0, :-1, best_y, best_x]
            max_known_class_score = known_logits_at_cell.max().item()
            ood_gating = 1.0 - max_known_class_score
            
            # Calculate entropy/uncertainty
            kl = torch.clamp(known_logits_at_cell, 1e-6, 1.0 - 1e-6)
            entropy = (-kl * torch.log(kl) - (1.0 - kl) * torch.log(1.0 - kl)).mean()
            p_un = entropy.item()
            
            # Calculate attribute score
            # In predict_unknown, self(img_feats, att_embeddings)[0] returns attribute logits
            unknown_predictions = model.bbox_head(img_feats, att_feats)[0]
            unknown_logits_at_cell = unknown_predictions[best_level][0, :, best_y, best_x].sigmoid()
            p_b = model.bbox_head.compute_weighted_top_k_attributes(
                unknown_logits_at_cell.unsqueeze(0), 
                k=model.bbox_head.top_k
            ).item()
            
            print("\n" + "-"*50)
            print("         HAUF UNKNOWN ANOMALY SCORE ANALYSIS        ")
            print("-"*50)
            print(f"-> Top-K Attribute Score (P_b):          {p_b:.4f}")
            print(f"-> Known Class Uncertainty (P_un):      {p_un:.4f}")
            print(f"-> Out-of-Distribution Gating (1-max_C): {ood_gating:.4f}")
            print(f"-> Final Calculated Unknown Score (P_u):  {max_pu:.4f}")
            print("-"*50)
            
            return max_pu
            
    except Exception as e:
        print(f"Warning: Failed to extract custom HAUF components: {e}")
        return 0.0


def main():
    args = parse_args()
    print("="*60)
    print("      IP102 TWO-STAGE IMAGE RETRIEVAL INFERENCE      ")
    print("="*60)

    # 1. Load Gallery Index
    if not os.path.exists(args.gallery_index):
        print(f"Error: Gallery index file not found at {args.gallery_index}. Please run build_gallery_index.py first.")
        return
        
    with open(args.gallery_index, "rb") as f:
        gallery_records = pickle.load(f)
    print(f"-> Loaded {len(gallery_records)} gallery records from {args.gallery_index}")

    # Resolve model and backward compatibility
    if args.extractor_model is None:
        if args.clip_model is not None:
            args.extractor_model = args.clip_model
        else:
            if args.extractor_type == "clip":
                args.extractor_model = "openai/clip-vit-base-patch32"
            elif args.extractor_type == "dinov2":
                args.extractor_model = "facebook/dinov2-base"
            else:
                args.extractor_model = "google/vit-base-patch16-224"

    # 2. Load OW-OVD Detector Model
    print(f"-> Initializing Detector model on {args.device}...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    # 3. Load feature extractor model
    extractor_model, extractor_processor = load_feature_extractor(
        model_name=args.extractor_model,
        extractor_type=args.extractor_type,
        device=args.device
    )

    # 4. Process Query Image
    print(f"-> Loading query image: {args.query_image}")
    query_img_pil = Image.open(args.query_image).convert("RGB")
    width, height = query_img_pil.size

    # Stage 1: Run OW-OVD Detector
    print("-> Stage 1: Detecting pests in query image...")
    img_bgr = cv2.cvtColor(np.array(query_img_pil), cv2.COLOR_RGB2BGR)
    det_result = inference_detector(model, img_bgr)
    
    pred_instances = det_result.pred_instances
    boxes = pred_instances.bboxes.cpu().numpy()  # (N, 4) xyxy
    scores = pred_instances.scores.cpu().numpy()  # (N,)
    
    best_box = None
    if len(scores) > 0:
        best_idx = np.argmax(scores)
        if scores[best_idx] >= args.score_thr:
            best_box = boxes[best_idx]
            print(f"-> Pest detected with confidence score: {scores[best_idx]:.4f}")

    # Apply cropping with 10% padding
    is_cropped = False
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
        
        cropped_img = query_img_pil.crop((x1_pad, y1_pad, x2_pad, y2_pad))
        is_cropped = True
        print(f"-> Cropped pest Region of Interest: [{x1_pad}, {y1_pad}, {x2_pad}, {y2_pad}]")
    else:
        # Fallback
        cropped_img = query_img_pil
        print("-> Fallback: No pest detected above threshold. Using the full image.")

    # Open-World Safeguard: HAUF Anomaly Check
    if is_cropped:
        p_u = extract_hauf_safeguard(model, cropped_img, args.device)
        if p_u > args.anomaly_thr:
            print(f"\n⚠️ WARNING: Potential unknown/novel pest detected! Species might not exist in the database. (P_u: {p_u:.4f} > {args.anomaly_thr})")
        else:
            print(f"-> Anomaly score (P_u = {p_u:.4f}) is below threshold. Pest considered a known species.")

    # Stage 2: Extract visual embedding using selected extractor
    with torch.no_grad():
        features = extract_visual_features(
            model=extractor_model,
            processor=extractor_processor,
            images=[cropped_img],
            extractor_type=args.extractor_type,
            device=args.device
        )
        query_embedding = features.cpu().numpy()[0]

    # Stage 3: Match and rank gallery
    gallery_embeddings = np.array([rec["feature_vector"] for rec in gallery_records])
    
    # Compute Cosine Similarity
    similarities = np.dot(query_embedding, gallery_embeddings.T)
    
    # Sort
    top_indices = np.argsort(similarities)[::-1][:args.top_k]
    top_scores = similarities[top_indices]
    
    print("\n" + "="*50)
    print(f"           TOP-{args.top_k} RETRIEVED PEST IMAGES           ")
    print("="*50)
    for idx, (g_idx, score) in enumerate(zip(top_indices, top_scores), 1):
        rec = gallery_records[g_idx]
        print(f"{idx:<2}. Score: {score:.4f} | Label: {rec['class_label']:<30} | Path: {os.path.basename(rec['image_path'])}")
    print("="*50)

    # 5. Visualizer and output generator
    fig = plt.figure(figsize=(15, 8))
    
    # Plot Left: Query image + Highlight box
    ax_query = fig.add_subplot(2, 3, 1)
    query_show = query_img_pil.copy()
    if is_cropped and best_box is not None:
        draw = ImageDraw.Draw(query_show)
        draw.rectangle(best_box.tolist(), outline="yellow", width=4)
        draw.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], outline="red", width=2)
        
    ax_query.imshow(query_show)
    ax_query.axis("off")
    title_str = "Query Image\n(Yellow = Detector Box, Red = Pad Crop)" if is_cropped else "Query Image (Fallback)"
    ax_query.set_title(title_str, fontsize=12, fontweight="bold")
    
    # Plot Right: Retrieval Grid (Top 5)
    # We display up to 5 images on the right / bottom row
    num_to_plot = min(5, args.top_k)
    for i in range(num_to_plot):
        g_idx = top_indices[i]
        rec = gallery_records[g_idx]
        score = top_scores[i]
        
        ax_ret = fig.add_subplot(2, 3, i + 2)
        try:
            ret_img = Image.open(rec["image_path"])
            ax_ret.imshow(ret_img)
        except Exception:
            ax_ret.text(0.5, 0.5, "Image Load Error", ha="center")
            
        ax_ret.axis("off")
        ax_ret.set_title(f"Rank {i+1} (Score: {score:.3f})\n{rec['class_label']}", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"-> Saved visualization grid to: {args.output}")


if __name__ == "__main__":
    main()
