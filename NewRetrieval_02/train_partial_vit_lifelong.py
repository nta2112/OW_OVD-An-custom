import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple

# Add parent directory of this script to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# MMCV/MMENGINE spoofing and patches to ensure compatibility on Kaggle
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

    # Spoof mmcv version
    try:
        import mmcv
        mmcv.__version__ = '2.0.1'
    except Exception:
        pass

    # Monkey patch Transformers check_torch_load_is_safe
    try:
        import transformers.utils.import_utils as transformers_import_utils
        transformers_import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    except Exception:
        pass

patch_environment()

from image_retrieval import load_feature_extractor
from train_mlp_adapter_lifelong import TASK_GROUPS, load_train_crops, batch_hard_triplet_loss

class CropsDataset(Dataset):
    def __init__(self, crops_info: List[Tuple[str, list, int]]):
        self.crops_info = crops_info
        
    def __len__(self):
        return len(self.crops_info)
        
    def __getitem__(self, idx):
        img_path, bbox, label = self.crops_info[idx]
        try:
            img_pil = Image.open(img_path).convert("RGB")
            w_img, h_img = img_pil.size
            x, y, w, h = bbox
            
            # Add padding
            pad_w = int(0.1 * w)
            pad_h = int(0.1 * h)
            x1 = max(0, int(x - pad_w))
            y1 = max(0, int(y - pad_h))
            x2 = min(w_img, int(x + w + pad_w))
            y2 = min(h_img, int(y + h + pad_h))
            
            crop = img_pil.crop((x1, y1, x2, y2))
            if crop.size[0] <= 0 or crop.size[1] <= 0:
                crop = Image.new("RGB", (224, 224), (128, 128, 128))
        except Exception:
            crop = Image.new("RGB", (224, 224), (128, 128, 128))
            
        return crop, label

def make_collate_fn(processor, extractor_type: str):
    def collate_fn(batch):
        crops = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        if extractor_type == "clip":
            inputs = processor(images=crops, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=crops, return_tensors="pt")
            
        return inputs, torch.tensor(labels, dtype=torch.long)
    return collate_fn

def unfreeze_last_blocks(model, extractor_type: str, unfreeze_blocks: int = 2):
    """
    Freezes the entire backbone except the last N blocks of the Transformer encoder.
    """
    # 1. Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze specific parts depending on the model structure
    if extractor_type == "clip":
        # CLIP image model has visual encoder inside vision_model
        vision_model = model.vision_model
        # Unfreeze post_layernorm and visual_projection
        if hasattr(model, "visual_projection"):
            for param in model.visual_projection.parameters():
                param.requires_grad = True
        for param in vision_model.post_layernorm.parameters():
            param.requires_grad = True
            
        # Unfreeze last N layers
        layers = vision_model.encoder.layers
        num_layers = len(layers)
        for i in range(max(0, num_layers - unfreeze_blocks), num_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
                
    elif extractor_type in ["dinov2", "vit"]:
        # HuggingFace ViT or DINOv2
        # Unfreeze layernorm
        if hasattr(model, "layernorm"):
            for param in model.layernorm.parameters():
                param.requires_grad = True
        
        # Unfreeze last N layers
        layers = model.encoder.layer
        num_layers = len(layers)
        for i in range(max(0, num_layers - unfreeze_blocks), num_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
                
    print(f"-> Set requires_grad=True for the last {unfreeze_blocks} block(s) of {extractor_type} backbone.")

def extract_student_features(model, inputs, extractor_type: str, device: str) -> torch.Tensor:
    # Prepare batch inputs
    batch_inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if extractor_type == "clip":
        features = model.get_image_features(**batch_inputs)
    else:
        outputs = model(**batch_inputs)
        if extractor_type == "dinov2":
            features = outputs.last_hidden_state[:, 0, :]
        else:
            features = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
            
    return features

def main():
    parser = argparse.ArgumentParser(description="Train Lifelong Partial ViT (Backbone Fine-Tuning) for CBIR")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to IP102 dataset root")
    parser.add_argument("--current-task", type=int, required=True, choices=[1, 2, 3, 4], help="Task index to train (1-4)")
    parser.add_argument("--extractor-type", type=str, default="vit", choices=["clip", "dinov2", "vit"], help="Feature extractor type")
    parser.add_argument("--extractor-model", type=str, required=True, help="Model name or local path for frozen extractor")
    parser.add_argument("--output-dir", type=str, default="work_dirs/partial_vit", help="Directory to save checkpoint files")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (should be small for backbone fine-tuning)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="Number of last blocks to unfreeze in ViT/DINOv2")
    parser.add_argument("--triplet-margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--distill-weight", type=float, default=1.0, help="Weight for distillation loss")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Determine Model Path & Teacher Model Path
    # For task > 1, load previous task checkpoint as starting weights and teacher weights
    prev_task = args.current_task - 1
    prev_task_dir = os.path.join(args.output_dir, f"task_{prev_task}")
    
    model_name_or_path = args.extractor_model
    teacher_model_path = None
    
    if args.current_task > 1:
        if os.path.exists(prev_task_dir):
            model_name_or_path = prev_task_dir
            teacher_model_path = prev_task_dir
            print(f"-> Continuing continual training from previous task checkpoint: {prev_task_dir}")
        else:
            print(f"⚠️ Warning: Previous task checkpoint {prev_task_dir} not found! Starting from raw base model.")
            
    # 2. Load Models
    student_model, extractor_processor = load_feature_extractor(model_name_or_path, args.extractor_type, args.device)
    student_model.train()
    
    # Freeze all layers except the last blocks
    unfreeze_last_blocks(student_model, args.extractor_type, args.unfreeze_blocks)
    
    teacher_model = None
    if teacher_model_path is not None:
        teacher_model, _ = load_feature_extractor(teacher_model_path, args.extractor_type, args.device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
            
    # 3. Load Crops Dataset
    target_classes = TASK_GROUPS[args.current_task]
    crops_info = load_train_crops(args.dataset_root, target_classes)
    
    if not crops_info:
        print(f"Error: No training annotations found for classes {target_classes}!")
        return
        
    dataset = CropsDataset(crops_info)
    collate_fn = make_collate_fn(extractor_processor, args.extractor_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # 4. Optimizer setup
    params_to_update = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=1e-4)
    
    # 5. Training Loop
    print(f"-> Starting training for Task {args.current_task} ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        student_model.train()
        epoch_loss = 0.0
        epoch_triplet = 0.0
        epoch_distill = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for inputs, labels in loop:
            optimizer.zero_grad()
            
            # Forward student features
            features = extract_student_features(student_model, inputs, args.extractor_type, args.device)
            features_norm = F.normalize(features, p=2, dim=1)
            
            # Compute Batch-Hard Triplet Loss
            triplet_loss = batch_hard_triplet_loss(features_norm, labels.to(args.device), margin=args.triplet_margin)
            
            # Compute Distillation Loss
            distill_loss = torch.tensor(0.0, device=args.device)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_features = extract_student_features(teacher_model, inputs, args.extractor_type, args.device)
                    teacher_features_norm = F.normalize(teacher_features, p=2, dim=1)
                distill_loss = F.mse_loss(features_norm, teacher_features_norm)
                
            loss = triplet_loss + args.distill_weight * distill_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_triplet += triplet_loss.item()
            epoch_distill += distill_loss.item()
            
            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "triplet": f"{triplet_loss.item():.4f}",
                "dist": f"{distill_loss.item():.4f}"
            })
            
        avg_loss = epoch_loss / len(dataloader)
        avg_triplet = epoch_triplet / len(dataloader)
        avg_distill = epoch_distill / len(dataloader)
        print(f"Epoch {epoch}/{args.epochs} Summary | Loss: {avg_loss:.4f} (Triplet: {avg_triplet:.4f}, Distill: {avg_distill:.4f})")
        
    # 6. Save Fine-Tuned Model Checkpoint (Full HuggingFace format)
    save_path = os.path.join(args.output_dir, f"task_{args.current_task}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save model and processor config
    student_model.save_pretrained(save_path)
    extractor_processor.save_pretrained(save_path)
    print(f"====== Saved Task {args.current_task} Fine-tuned ViT backbone to: {save_path} ======\n")

if __name__ == "__main__":
    main()
