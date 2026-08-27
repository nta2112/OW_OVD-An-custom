"""
Image Retrieval (CBIR) Utilities
================================
This script implements a content-based image retrieval (CBIR) pipeline using
a pretrained CLIP vision encoder. It replaces bounding box detection with 
image similarity search (top-5 / top-10 retrieval) and evaluates Recall@K.

Features:
- Locates IP102 dataset directories automatically.
- Extracts image paths and category labels from COCO JSON annotations.
- Uses CLIP (Vision Encoder) to extract normalized feature embeddings.
- Computes Cosine Similarity to find top-K similar images.
- Evaluates Recall@1, Recall@5, Recall@10 performance per class.
"""

import os
import json
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel


def find_dataset_root() -> str:
    """
    Locates the IP102 dataset directory by checking common local and Kaggle paths.
    """
    search_paths = [
        '/kaggle/input/datasets/nta212/ip102-for-object-detection',
        '/kaggle/input/ip102-for-object-detection',
        'data/IP102',
        '.'
    ]
    
    # First search direct paths
    for path in search_paths:
        if os.path.exists(os.path.join(path, 'train.json')):
            print(f"-> Found dataset root at: {path}")
            return path
            
    # Fallback to search recursively for train.json
    fallback_paths = glob.glob('/kaggle/input/**/train.json', recursive=True)
    if fallback_paths:
        path = os.path.dirname(fallback_paths[0])
        print(f"-> Found dataset root recursively at: {path}")
        return path
        
    print("-> WARNING: Could not find train.json. Defaulting to current directory '.'")
    return '.'


def collect_images(dataset_root: str, split: str = 'test') -> List[Dict]:
    """
    Collects image file paths and their associated class category labels.
    Parses COCO JSON annotations (e.g., train.json, test.json, val.json).
    
    Returns:
        List of dicts: [
            {"image_path": "...", "class_id": 1, "class_name": "..."},
            ...
        ]
    """
    json_path = os.path.join(dataset_root, f'{split}.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found at: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
        
    # Map category ID to category name
    categories = coco_data.get('categories', [])
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # Map image ID to file name and image path
    images = coco_data.get('images', [])
    image_id_to_info = {}
    
    # Find the folder containing images. Sometimes images are in 'images/', 'test/', or the root itself.
    image_folder = dataset_root
    for subfolder in [split, 'images', 'test/test', 'train/train']:
        test_path = os.path.join(dataset_root, subfolder)
        if os.path.exists(test_path) and os.path.isdir(test_path):
            image_folder = test_path
            break
            
    for img in images:
        file_name = os.path.basename(img['file_name'])
        image_id_to_info[img['id']] = {
            'file_name': file_name,
            'image_path': os.path.join(image_folder, file_name)
        }
        
    # Map annotations to images to extract class label
    # In object detection, an image might have multiple boxes. We assign the first box's category as the main class label.
    annotations = coco_data.get('annotations', [])
    image_to_class = {}
    for ann in annotations:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if img_id not in image_to_class:
            image_to_class[img_id] = cat_id
            
    # Gather dataset items
    dataset_items = []
    for img_id, info in image_id_to_info.items():
        class_id = image_to_class.get(img_id, -1)
        if class_id == -1:
            # Skip if there's no class annotated
            continue
            
        class_name = cat_id_to_name.get(class_id, f"unknown_class_{class_id}")
        
        # Verify physical file existence
        # Sometimes paths are nested, let's find the correct file
        real_path = info['image_path']
        if not os.path.exists(real_path):
            # Try searching in dataset root recursively
            basename = os.path.basename(real_path)
            searched_files = glob.glob(os.path.join(dataset_root, '**', basename), recursive=True)
            if searched_files:
                real_path = searched_files[0]
            else:
                continue # Skip if file doesn't exist
                
        dataset_items.append({
            "image_path": real_path,
            "class_id": class_id,
            "class_name": class_name
        })
        
    print(f"-> Collected {len(dataset_items)} images for split '{split}' from {dataset_root}")
    return dataset_items


def load_feature_extractor(model_name: str, extractor_type: str = "clip", device: str = "cuda") -> Tuple[torch.nn.Module, object]:
    """
    Loads pretrained feature extractor model and processor from HuggingFace.
    extractor_type: 'clip', 'dinov2', or 'vit'
    """
    from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
    print(f"-> Loading feature extractor ({extractor_type}): {model_name} on {device}...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    if extractor_type == "clip":
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        processor = CLIPProcessor.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        
    model.to(device)
    model.eval()
    return model, processor


def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda") -> Tuple[torch.nn.Module, object]:
    """
    Backward compatibility wrapper to load CLIP model.
    """
    return load_feature_extractor(model_name, extractor_type="clip", device=device)


def extract_visual_features(model: torch.nn.Module, processor: object, images: List[Image.Image], extractor_type: str = "clip", device: str = "cuda") -> torch.Tensor:
    """
    Extracts L2-normalized visual feature embeddings for a batch of PIL images.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if extractor_type == "clip":
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        features = model.get_image_features(**inputs)
        # Unpack if returned as BaseModelOutputWithPooling
        if not isinstance(features, torch.Tensor):
            if hasattr(features, "pooler_output") and features.pooler_output is not None:
                features = features.pooler_output
            elif hasattr(features, "image_embeds") and features.image_embeds is not None:
                features = features.image_embeds
            elif isinstance(features, (list, tuple)):
                features = features[0]
    else:
        # DinoV2 or standard ViT
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        if extractor_type == "dinov2":
            features = outputs.last_hidden_state[:, 0, :]
        else:
            features = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
            
    # L2 normalize
    features = features / features.norm(dim=-1, keepdim=True)
    return features


def extract_embeddings(
    dataset_items: List[Dict], 
    model: torch.nn.Module, 
    processor: object, 
    extractor_type: str = "clip",
    device: str = "cuda", 
    batch_size: int = 32
) -> np.ndarray:
    """
    Extracts L2-normalized visual feature embeddings for a list of images.
    
    Returns:
        np.ndarray of shape (N, D) where N is number of images, D is embedding dimension (e.g. 512)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    embeddings = []
    
    # Process images in batches
    for i in tqdm(range(0, len(dataset_items), batch_size), desc="Extracting embeddings"):
        batch = dataset_items[i:i+batch_size]
        images = []
        for item in batch:
            try:
                img = Image.open(item['image_path']).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error reading image {item['image_path']}: {e}")
                # Create a placeholder image to keep index alignment
                images.append(Image.new("RGB", (224, 224), color=0))
                
        # Preprocess and encode
        with torch.no_grad():
            features = extract_visual_features(model, processor, images, extractor_type=extractor_type, device=device)
            embeddings.append(features.cpu().numpy())
            
    return np.vstack(embeddings)


def build_index(embeddings: np.ndarray) -> np.ndarray:
    """
    Prepares the gallery embeddings. For simple cosine similarity matching, 
    this returns the normalized numpy array.
    """
    return embeddings


def retrieve(
    query_embedding: np.ndarray, 
    gallery_embeddings: np.ndarray, 
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes cosine similarity between the query embedding and the gallery index.
    
    Args:
        query_embedding: array of shape (D,) or (1, D)
        gallery_embeddings: array of shape (M, D)
        top_k: number of similar images to retrieve
        
    Returns:
        similarities: array of shape (top_k,)
        indices: array of shape (top_k,)
    """
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
        
    # Since embeddings are L2 normalized, Cosine Similarity is simply dot product
    # Shape: (1, M)
    similarities = np.dot(query_embedding, gallery_embeddings.T)[0]
    
    # Sort indices in descending order of similarity
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    
    return top_scores, top_indices


def evaluate(
    query_items: List[Dict],
    query_embeddings: np.ndarray,
    gallery_items: List[Dict],
    gallery_embeddings: np.ndarray,
    top_k: int = 10
) -> Dict[str, Dict[int, float]]:
    """
    Evaluates Recall@K per class. 
    A query is successfully retrieved at Rank K if there is at least one image in the top-K
    retrieved gallery images that belongs to the same class as the query image.
    
    Returns:
        Dict mapping class_name -> {1: recall@1, 5: recall@5, 10: recall@10}
    """
    class_eval = {} # class_name -> {ranks: [success_list]}
    
    for q_idx, q_item in enumerate(tqdm(query_items, desc="Evaluating Queries")):
        q_class = q_item['class_id']
        q_class_name = q_item['class_name']
        
        # Get top matching indices in gallery
        _, top_g_indices = retrieve(query_embeddings[q_idx], gallery_embeddings, top_k=top_k)
        
        # Initialize metrics for this class if not present
        if q_class_name not in class_eval:
            class_eval[q_class_name] = {1: [], 5: [], 10: []}
            
        # Check if the query's class appears in top-K ranks
        # Note: We must exclude the query image itself if it happens to be in the gallery database!
        # Under normal CBIR settings, query images are not part of the gallery.
        g_classes = [gallery_items[idx]['class_id'] for idx in top_g_indices]
        
        # Check Recall@1
        rec1 = 1.0 if q_class in g_classes[:1] else 0.0
        # Check Recall@5
        rec5 = 1.0 if q_class in g_classes[:5] else 0.0
        # Check Recall@10
        rec10 = 1.0 if q_class in g_classes[:10] else 0.0
        
        class_eval[q_class_name][1].append(rec1)
        class_eval[q_class_name][5].append(rec5)
        class_eval[q_class_name][10].append(rec10)
        
    # Calculate average recalls per class
    class_reports = {}
    for cls_name, ranks_data in class_eval.items():
        class_reports[cls_name] = {
            1: np.mean(ranks_data[1]),
            5: np.mean(ranks_data[5]),
            10: np.mean(ranks_data[10]),
            'count': len(ranks_data[1])
        }
        
    return class_reports


def print_report(class_reports: Dict[str, Dict[str, float]]):
    """
    Prints a formatted evaluation report containing Recall@1, Recall@5, and Recall@10 metrics.
    """
    print("\n" + "="*70)
    print(f"{'Class Name':<35} | {'Count':<6} | {'Recall@1':<8} | {'Recall@5':<8} | {'Recall@10':<8}")
    print("="*70)
    
    total_q = 0
    macro_r1, macro_r5, macro_r10 = 0.0, 0.0, 0.0
    weighted_r1, weighted_r5, weighted_r10 = 0.0, 0.0, 0.0
    
    for cls_name, metrics in sorted(class_reports.items()):
        count = metrics['count']
        r1, r5, r10 = metrics[1], metrics[5], metrics[10]
        
        print(f"{cls_name:<35} | {count:<6} | {r1:<8.4f} | {r5:<8.4f} | {r10:<8.4f}")
        
        total_q += count
        macro_r1 += r1
        macro_r5 += r5
        macro_r10 += r10
        
        weighted_r1 += r1 * count
        weighted_r5 += r5 * count
        weighted_r10 += r10 * count
        
    num_classes = len(class_reports)
    if num_classes > 0:
        macro_r1 /= num_classes
        macro_r5 /= num_classes
        macro_r10 /= num_classes
        
        weighted_r1 /= total_q
        weighted_r5 /= total_q
        weighted_r10 /= total_q
        
    print("="*70)
    print(f"{'Macro Average':<35} | {num_classes:<6} | {macro_r1:<8.4f} | {macro_r5:<8.4f} | {macro_r10:<8.4f}")
    print(f"{'Micro/Weighted Average':<35} | {total_q:<6} | {weighted_r1:<8.4f} | {weighted_r5:<8.4f} | {weighted_r10:<8.4f}")
    print("="*70)


if __name__ == "__main__":
    # Test/Example Pipeline Run
    try:
        root = find_dataset_root()
        
        # We can split the collected test set into query and gallery, or treat
        # val set as query and test set as gallery.
        # For demonstration, we collect from 'test' and evaluate on split portions
        test_items = collect_images(root, split='test')
        
        if len(test_items) < 2:
            print("Not enough images found to run retrieval demo. Please check dataset path.")
        else:
            # Load CLIP Model
            model, processor = load_clip(model_name="openai/clip-vit-base-patch32", device="cuda")
            
            # Split items: First 50 as queries, rest as gallery database
            query_items = test_items[:50]
            gallery_items = test_items[50:]
            
            print(f"-> Extracting query embeddings ({len(query_items)} images)...")
            query_embeds = extract_embeddings(query_items, model, processor, device="cuda")
            
            print(f"-> Extracting gallery embeddings ({len(gallery_items)} images)...")
            gallery_embeds = extract_embeddings(gallery_items, model, processor, device="cuda")
            
            # Build Index
            gallery_index = build_index(gallery_embeds)
            
            # Evaluate
            reports = evaluate(query_items, query_embeds, gallery_items, gallery_index, top_k=10)
            print_report(reports)
            
    except Exception as e:
        print(f"Error running pipeline demo: {e}")
        print("Note: Make sure PyTorch and transformers library are installed correctly.")
