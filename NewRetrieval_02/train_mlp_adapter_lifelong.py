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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from image_retrieval import load_feature_extractor, extract_visual_features

# Task class boundaries (same as in evaluate_retrieval_lifelong.py)
TASK_GROUPS = {
    1: ['14', '15', '16', '18', '22', '23', '24'],       # Task 1 classes (7 classes)
    2: ['25', '26', '37', '38', '39', '45'],             # Task 2 classes (6 classes)
    3: ['46', '47', '48', '49', '50', '66'],             # Task 3 classes (6 classes)
    4: ['67', '69', '70', '86', '101']                  # Task 4 classes (5 classes)
}

class MLPAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class FeaturesDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Computes batch-hard triplet loss.
    For each anchor, finds the hardest positive and hardest negative in the batch.
    """
    # Pairwise distance matrix (Euclidean)
    dists = torch.cdist(embeddings, embeddings, p=2)
    
    # Masks for positives and negatives
    same_label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # Hardest positive: max distance among those with the same label (excluding self)
    # Set distance to self (diagonal) to 0, which is fine since same_label_mask contains it
    pos_dists = dists * same_label_mask.float()
    hardest_pos, _ = pos_dists.max(dim=1)
    
    # Hardest negative: min distance among those with different labels
    # Set distance to same-label pairs to a large value
    max_dist = dists.max()
    neg_dists = dists + same_label_mask.float() * (max_dist + 1e5)
    hardest_neg, _ = neg_dists.min(dim=1)
    
    # Loss
    losses = F.relu(hardest_pos - hardest_neg + margin)
    return losses.mean()

def load_train_crops(dataset_root: str, target_classes: List[str]) -> List[Tuple[str, int]]:
    """
    Loads train image crops for target classes using ground truth bounding boxes from train.json.
    """
    json_path = os.path.join(dataset_root, "train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file train.json not found in {dataset_root}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
        
    cat_name_to_id = {cat['name']: cat['id'] for cat in coco.get('categories', [])}
    target_cat_ids = {cat_name_to_id[name] for name in target_classes if name in cat_name_to_id}
    
    # Filter images containing annotations of target classes
    img_id_to_file = {img['id']: img['file_name'] for img in coco.get('images', [])}
    
    # Index all image paths in dataset_root recursively once
    print("-> Scanning dataset root to index image file paths...")
    image_path_map = {}
    for root, _, files in os.walk(dataset_root):
        for f_name in files:
            if f_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path_map[f_name] = os.path.join(root, f_name)
    print(f"-> Indexed {len(image_path_map)} images.")
    
    crops_info = []
    for ann in coco.get('annotations', []):
        cat_id = ann['category_id']
        if cat_id in target_cat_ids:
            img_id = ann['image_id']
            file_name = os.path.basename(img_id_to_file.get(img_id, ''))
            if not file_name:
                continue
            
            img_path = image_path_map.get(file_name)
            if img_path is None or not os.path.exists(img_path):
                continue
                    
            bbox = ann['bbox'] # [x, y, w, h]
            crops_info.append((img_path, bbox, cat_id))
            
    # Map target category IDs to unique integers from 0 to num_classes-1
    unique_cat_ids = sorted(list(target_cat_ids))
    cat_id_map = {cat_id: i for i, cat_id in enumerate(unique_cat_ids)}
    
    mapped_crops = []
    for img_path, bbox, cat_id in crops_info:
        mapped_crops.append((img_path, bbox, cat_id_map[cat_id]))
        
    print(f"-> Found {len(mapped_crops)} training annotations for classes {target_classes}")
    return mapped_crops

def extract_crops_features(crops_info: List[Tuple[str, list, int]], extractor_model, extractor_processor, extractor_type: str, device: str, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops training images using ground truth boxes and extracts feature vectors using frozen model.
    """
    device = torch.device(device)
    features_list = []
    labels_list = []
    
    crops = []
    labels_batch = []
    
    for img_path, bbox, label in tqdm(crops_info, desc="Extracting features from train crops"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            w_img, h_img = img_pil.size
            x, y, w, h = bbox
            
            # Add small padding
            pad_w = int(0.1 * w)
            pad_h = int(0.1 * h)
            x1 = max(0, int(x - pad_w))
            y1 = max(0, int(y - pad_h))
            x2 = min(w_img, int(x + w + pad_w))
            y2 = min(h_img, int(y + h + pad_h))
            
            crop = img_pil.crop((x1, y1, x2, y2))
            crops.append(crop)
            labels_batch.append(label)
            
            if len(crops) >= batch_size:
                with torch.no_grad():
                    feats = extract_visual_features(extractor_model, extractor_processor, crops, extractor_type, str(device))
                features_list.append(feats.cpu())
                labels_list.extend(labels_batch)
                crops = []
                labels_batch = []
        except Exception:
            continue
            
    if crops:
        with torch.no_grad():
            feats = extract_visual_features(extractor_model, extractor_processor, crops, extractor_type, str(device))
        features_list.append(feats.cpu())
        labels_list.extend(labels_batch)
        
    if not features_list:
        raise ValueError("No features were successfully extracted. Check image paths or GPU compatibility.")
        
    return torch.cat(features_list, dim=0), torch.tensor(labels_list, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Train Lifelong MLP Adapter for CBIR Retrieval")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to IP102 dataset root")
    parser.add_argument("--current-task", type=int, required=True, choices=[1, 2, 3, 4], help="Task index to train (1-4)")
    parser.add_argument("--extractor-type", type=str, default="vit", choices=["clip", "dinov2", "vit"], help="Frozen extractor type")
    parser.add_argument("--extractor-model", type=str, required=True, help="Model name or local path for frozen extractor")
    parser.add_argument("--output-dir", type=str, default="work_dirs/mlp_adapter", help="Directory to save checkpoint files")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--triplet-margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--distill-weight", type=float, default=1.0, help="Weight for distillation loss")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load frozen feature extractor
    extractor_model, extractor_processor = load_feature_extractor(args.extractor_model, args.extractor_type, args.device)
    
    # 2. Collect crops info and extract features
    target_classes = TASK_GROUPS[args.current_task]
    crops_info = load_train_crops(args.dataset_root, target_classes)
    
    if not crops_info:
        print(f"Error: No training annotations found for classes {target_classes}!")
        return
        
    features, labels = extract_crops_features(crops_info, extractor_model, extractor_processor, args.extractor_type, args.device, args.batch_size)
    print(f"Extracted feature matrix: {features.shape}, Labels: {labels.shape}")
    
    # Determine feature input dimension
    input_dim = features.shape[1]
    
    # 3. Initialize MLP Adapter model
    model = MLPAdapter(input_dim=input_dim, output_dim=256).to(args.device)
    
    prev_model = None
    if args.current_task > 1:
        prev_model_path = os.path.join(args.output_dir, f"mlp_adapter_task_{args.current_task-1}.pth")
        if os.path.exists(prev_model_path):
            print(f"-> Loading previous task weights from: {prev_model_path}")
            model.load_state_dict(torch.load(prev_model_path, map_location=args.device))
            
            # Load another frozen instance for distillation target
            prev_model = MLPAdapter(input_dim=input_dim, output_dim=256).to(args.device)
            prev_model.load_state_dict(torch.load(prev_model_path, map_location=args.device))
            prev_model.eval()
        else:
            print(f"Warning: Task {args.current_task} > 1 but previous checkpoint {prev_model_path} not found. Starting from scratch.")
            
    # 4. Prepare DataLoader
    dataset = FeaturesDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    print(f"Starting training MLP Adapter for Task {args.current_task}...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_metric = 0.0
        total_distill = 0.0
        
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(args.device)
            batch_labels = batch_labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Metric Loss: Triplet
            loss_metric = batch_hard_triplet_loss(outputs, batch_labels, args.triplet_margin)
            
            # Distillation Loss (if task > 1)
            loss_distill = torch.tensor(0.0, device=args.device)
            if prev_model is not None:
                with torch.no_grad():
                    prev_outputs = prev_model(batch_features)
                loss_distill = F.mse_loss(outputs, prev_outputs)
                
            loss = loss_metric + args.distill_weight * loss_distill
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_features)
            total_metric += loss_metric.item() * len(batch_features)
            total_distill += loss_distill.item() * len(batch_features)
            
        avg_loss = total_loss / len(dataset)
        avg_metric = total_metric / len(dataset)
        avg_distill = total_distill / len(dataset)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} (Metric: {avg_metric:.4f}, Distill: {avg_distill:.4f})")
        
    # 6. Save model checkpoint
    output_path = os.path.join(args.output_dir, f"mlp_adapter_task_{args.current_task}.pth")
    torch.save(model.state_dict(), output_path)
    print(f"====== Saved Task {args.current_task} MLP Adapter checkpoint to: {output_path} ======\n")

if __name__ == "__main__":
    main()
