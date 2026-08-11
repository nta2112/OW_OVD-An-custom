import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Sequence, Dict
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import multi_apply
from mmyolo.registry import MODELS
from yolo_world.models.dense_heads.our_head_new import OurHeadModule, OurHead

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        embeddings: Tensor of shape (N, D) where N is the number of positive samples
        labels: Tensor of shape (N,) containing class IDs
        """
        if len(embeddings) < 3:
            return embeddings.new_tensor(0.0, requires_grad=True)
            
        # L2 distances between all positive embeddings
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        distances = torch.clamp(distances, min=0.0)
        
        # Mask for positive pairs (same label, different indices)
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        mask_pos = label_equal & ~indices_equal
        
        # Mask for negative pairs (different labels)
        mask_neg = ~label_equal
        
        triplet_loss = []
        N = len(embeddings)
        for i in range(N):
            pos_indices = torch.where(mask_pos[i])[0]
            neg_indices = torch.where(mask_neg[i])[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            d_ap = distances[i, pos_indices].unsqueeze(1) # (P, 1)
            d_an = distances[i, neg_indices].unsqueeze(0) # (1, N_neg)
            
            loss_temp = d_ap - d_an + self.margin
            loss_temp = torch.clamp(loss_temp, min=0.0)
            
            triplet_loss.append(loss_temp.mean())
            
        if len(triplet_loss) == 0:
            return embeddings.new_tensor(0.0, requires_grad=True)
            
        return torch.stack(triplet_loss).mean()

class AssignerWrapper:
    def __init__(self, original_assigner):
        self.original_assigner = original_assigner
        self.latest_result = None
        
    def __call__(self, *args, **kwargs):
        res = self.original_assigner(*args, **kwargs)
        self.latest_result = res
        return res

@MODELS.register_module()
class OurHeadRetrievalModule(OurHeadModule):
    def __init__(self, *args, retrieval_dim=256, **kwargs) -> None:
        self.retrieval_dim = retrieval_dim
        super().__init__(*args, **kwargs)
        
    def _init_layers(self) -> None:
        super()._init_layers()
        self.ret_preds = nn.ModuleList()
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        for i in range(self.num_levels):
            self.ret_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.retrieval_dim,
                              kernel_size=1)
                )
            )
            
    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           self.cls_preds, self.reg_preds, self.cls_contrasts, self.ret_preds)
                           
    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList, ret_pred: nn.ModuleList) -> Tuple:
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        
        # Retrieval embedding projection and normalization
        ret_embed = ret_pred(img_feat)
        ret_embed = F.normalize(ret_embed, p=2, dim=1) # shape: (b, retrieval_dim, h, w)
        
        if self.reg_max > 1:
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
            
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, ret_embed
        else:
            return cls_logit, bbox_preds, ret_embed

@MODELS.register_module()
class OurHeadRetrieval(OurHead):
    def __init__(self, *args, loss_retrieval_weight=0.5, retrieval_dim=256, triplet_margin=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_retrieval_weight = loss_retrieval_weight
        self.retrieval_dim = retrieval_dim
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        
        if hasattr(self, 'assigner') and self.assigner is not None:
            self.assigner = AssignerWrapper(self.assigner)
            
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        # Dynamically wrap assigner if it wasn't available at initialization
        if hasattr(self, 'assigner') and self.assigner is not None:
            if not isinstance(self.assigner, AssignerWrapper):
                self.assigner = AssignerWrapper(self.assigner)
                
        # x contains (cls_scores, bbox_preds, bbox_dist_preds, ret_embeds)
        cls_scores, bbox_preds, bbox_dist_preds, ret_embeds = x
        
        # Pass detection outputs to the parent class's loss logic
        det_x = (cls_scores, bbox_preds, bbox_dist_preds)
        losses = super().loss(det_x, batch_data_samples)
        
        # Capture TaskAlignedAssigner results
        if not hasattr(self, 'assigner') or self.assigner is None or self.assigner.latest_result is None:
            losses['loss_retrieval'] = ret_embeds[0].new_tensor(0.0, requires_grad=True)
            return losses
            
        assign_result = self.assigner.latest_result
        if isinstance(assign_result, list):
            gt_inds_list = [res.gt_inds for res in assign_result]
        else:
            gt_inds_list = assign_result.gt_inds
            if len(gt_inds_list.shape) == 2:
                gt_inds_list = [gt_inds_list[i] for i in range(gt_inds_list.shape[0])]
            else:
                gt_inds_list = [gt_inds_list]
                
        pos_embeddings_list = []
        pos_labels_list = []
        
        # Flatten ret_embeds to align with total anchors
        flatten_ret_embeds = []
        for ret_embed in ret_embeds:
            b, c, h, w = ret_embed.shape
            flat = ret_embed.view(b, c, -1).permute(0, 2, 1)
            flatten_ret_embeds.append(flat)
        flatten_ret_embeds = torch.cat(flatten_ret_embeds, dim=1) # (batch_size, total_anchors, retrieval_dim)
        
        for i, gt_inds in enumerate(gt_inds_list):
            fg_mask = gt_inds > 0
            if not fg_mask.any():
                continue
                
            img_pos_embeds = flatten_ret_embeds[i][fg_mask]
            gt_instances = batch_data_samples[i].gt_instances
            gt_labels = gt_instances.labels
            
            pos_gt_indices = gt_inds[fg_mask] - 1
            img_pos_labels = gt_labels[pos_gt_indices]
            
            pos_embeddings_list.append(img_pos_embeds)
            pos_labels_list.append(img_pos_labels)
            
        if len(pos_embeddings_list) > 0:
            all_pos_embeds = torch.cat(pos_embeddings_list, dim=0)
            all_pos_labels = torch.cat(pos_labels_list, dim=0)
            loss_triplet = self.triplet_loss(all_pos_embeds, all_pos_labels)
        else:
            loss_triplet = ret_embeds[0].new_tensor(0.0, requires_grad=True)
            
        losses['loss_retrieval'] = loss_triplet * self.loss_retrieval_weight
        return losses
