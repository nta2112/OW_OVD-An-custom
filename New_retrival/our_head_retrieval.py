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
from mmyolo.models.utils import gt_instances_preprocess

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

class AssignerWrapper(nn.Module):
    def __init__(self, original_assigner):
        super().__init__()
        self.original_assigner = original_assigner
        self.latest_result = None
        
    def forward(self, *args, **kwargs):
        res = self.original_assigner(*args, **kwargs)
        self.latest_result = res
        return res
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_assigner, name)


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
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
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
            
    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict], fusion_att: bool=False) -> dict:
        # Dynamically wrap assigner if it wasn't available at initialization
        if hasattr(self, 'assigner') and self.assigner is not None:
            if not isinstance(self.assigner, AssignerWrapper):
                self.assigner = AssignerWrapper(self.assigner)
                
        # Forward pass of OurHeadRetrievalModule (returns 4 lists during training)
        outs = self(img_feats, txt_feats)
        cls_scores, bbox_preds, bbox_dist_preds, ret_embeds = outs
        
        # Pass detection outputs
        det_outs = (cls_scores, bbox_preds, bbox_dist_preds)
        
        if self.att_embeddings is None:
            loss_inputs = det_outs + (None, batch_data_samples['bboxes_labels'],
                                      batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)
        else:
            if fusion_att: 
                num_att = self.att_embeddings.shape[0]
                att_feats = txt_feats[:, -num_att: , :]
                txt_feats = txt_feats[:, :-num_att, :]
            else:
                att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)
            
            with torch.no_grad():
                att_outs = self(img_feats, att_feats)[0]
                
            loss_inputs = det_outs + (att_outs, batch_data_samples['bboxes_labels'],
                                      batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)
            
        # Retrieval Loss (Triplet Loss)
        if not hasattr(self, 'assigner') or self.assigner is None or self.assigner.latest_result is None:
            losses['loss_retrieval'] = ret_embeds[0].new_tensor(0.0, requires_grad=True)
            return losses
            
        assign_result = self.assigner.latest_result
        
        # Flatten ret_embeds to align with total anchors
        flatten_ret_embeds = []
        for ret_embed in ret_embeds:
            b, c, h, w = ret_embed.shape
            flat = ret_embed.view(b, c, -1).permute(0, 2, 1)
            flatten_ret_embeds.append(flat)
        flatten_ret_embeds = torch.cat(flatten_ret_embeds, dim=1) # (batch_size, total_anchors, retrieval_dim)
        
        pos_embeddings_list = []
        pos_labels_list = []
        
        # Handle dict format (e.g. BatchTaskAlignedAssigner) vs standard list/class format
        if isinstance(assign_result, dict):
            fg_mask_pre_prior = assign_result['fg_mask_pre_prior']
            assigned_bboxes = assign_result['assigned_bboxes']
            
            # Prepare gt_info using gt_instances_preprocess
            if isinstance(batch_data_samples, dict):
                gt_info = gt_instances_preprocess(batch_data_samples['bboxes_labels'], fg_mask_pre_prior.shape[0])
            else:
                batch_gt_instances = [sample.gt_instances for sample in batch_data_samples]
                gt_info = gt_instances_preprocess(batch_gt_instances, fg_mask_pre_prior.shape[0])
                
            for i in range(fg_mask_pre_prior.shape[0]):
                fg_mask = fg_mask_pre_prior[i]
                if not fg_mask.any():
                    continue
                    
                img_pos_embeds = flatten_ret_embeds[i][fg_mask]
                
                img_gt_boxes = gt_info[i, :, 1:].to(assigned_bboxes.device, dtype=assigned_bboxes.dtype)
                img_gt_labels = gt_info[i, :, 0].long().to(assigned_bboxes.device)
                
                valid_gt_mask = img_gt_boxes.sum(dim=-1) > 0
                img_gt_boxes = img_gt_boxes[valid_gt_mask]
                img_gt_labels = img_gt_labels[valid_gt_mask]
                
                img_pos_assigned_boxes = assigned_bboxes[i][fg_mask]
                
                # Match assigned boxes to ground truth boxes to get indices
                dists = torch.abs(img_pos_assigned_boxes.unsqueeze(1) - img_gt_boxes.unsqueeze(0)).sum(dim=-1)
                pos_gt_indices = torch.argmin(dists, dim=-1)
                
                img_pos_labels = img_gt_labels[pos_gt_indices]
                
                pos_embeddings_list.append(img_pos_embeds)
                pos_labels_list.append(img_pos_labels)
        else:
            # Original list/class format
            if isinstance(assign_result, list):
                gt_inds_list = [res.gt_inds for res in assign_result]
            else:
                gt_inds_list = assign_result.gt_inds
                if len(gt_inds_list.shape) == 2:
                    gt_inds_list = [gt_inds_list[i] for i in range(gt_inds_list.shape[0])]
                else:
                    gt_inds_list = [gt_inds_list]
                    
            # Prepare gt_info using gt_instances_preprocess
            if isinstance(batch_data_samples, dict):
                gt_info = gt_instances_preprocess(batch_data_samples['bboxes_labels'], len(gt_inds_list))
            else:
                batch_gt_instances = [sample.gt_instances for sample in batch_data_samples]
                gt_info = gt_instances_preprocess(batch_gt_instances, len(gt_inds_list))
                
            for i, gt_inds in enumerate(gt_inds_list):
                fg_mask = gt_inds > 0
                if not fg_mask.any():
                    continue
                    
                img_pos_embeds = flatten_ret_embeds[i][fg_mask]
                
                img_gt_labels = gt_info[i, :, 0].long().to(gt_inds.device)
                
                pos_gt_indices = gt_inds[fg_mask] - 1
                img_pos_labels = img_gt_labels[pos_gt_indices]
                
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

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False, 
                fusion_att: bool = False) -> InstanceList:
        
        # Forward pass in evaluation/predict mode (returns 3 lists: cls_scores, bbox_preds, ret_embeds)
        x = self(img_feats, txt_feats)
        cls_scores, bbox_preds, ret_embeds = x
        
        # Pack only detection outputs
        det_outs = (cls_scores, bbox_preds)
        
        if self.att_embeddings.shape[0] != 25 * (self.num_classes):
            self.select_att()
            
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        
        if self.att_embeddings is None:
            predictions = self.predict_by_feat(*det_outs,
                                               batch_img_metas=batch_img_metas,
                                               rescale=rescale)
        else:
            if fusion_att: 
                num_att = self.att_embeddings.shape[0]
                att_feats = txt_feats[:, -num_att: , :]
                txt_feats = txt_feats[:, :-num_att, :]
            else:
                if self.attr_sel_for_known_only:
                    att_feats = self.all_atts[None].repeat(txt_feats.shape[0], 1, 1)
                else:
                    att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)
            
            det_outs = self.predict_unknown(det_outs, img_feats, att_feats)
            predictions = self.predict_by_feat(*det_outs,
                                               batch_img_metas=batch_img_metas,
                                               rescale=rescale)
            
        # Extract retrieval embeddings for the predicted bounding boxes
        import torchvision.ops as tv_ops
        for i, pred in enumerate(predictions):
            if len(pred) == 0:
                pred.features = pred.bboxes.new_zeros((0, self.retrieval_dim))
                continue
                
            bboxes = pred.bboxes
            img_meta = batch_img_metas[i]
            scale_factor = img_meta.get('scale_factor', (1.0, 1.0))
            if isinstance(scale_factor, float):
                scale_factor = (scale_factor, scale_factor)
                
            if rescale:
                w_scale, h_scale = scale_factor
                scaled_bboxes = bboxes.clone()
                scaled_bboxes[:, 0] /= w_scale
                scaled_bboxes[:, 2] /= w_scale
                scaled_bboxes[:, 1] /= h_scale
                scaled_bboxes[:, 3] /= h_scale
            else:
                scaled_bboxes = bboxes
                
            pooled_feats = []
            for level_idx, stride in enumerate([8, 16, 32]):
                feat = ret_embeds[level_idx][i:i+1] # (1, retrieval_dim, h_j, w_j)
                lvl_boxes = scaled_bboxes / stride
                rois = torch.cat([lvl_boxes.new_zeros(len(lvl_boxes), 1), lvl_boxes], dim=1)
                pooled = tv_ops.roi_align(feat, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
                pooled_feats.append(pooled.view(len(lvl_boxes), -1))
                
            img_box_embeds = torch.stack(pooled_feats, dim=0).mean(dim=0)
            img_box_embeds = F.normalize(img_box_embeds, p=2, dim=1)
            pred.features = img_box_embeds
            
        return predictions

