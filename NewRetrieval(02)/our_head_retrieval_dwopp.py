import math
import copy
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

class EpisodicHardMiningLoss(nn.Module):
    """
    Episodic Hard-mining Metric Loss (L_eps).
    Implements a Triplet Loss with Batch-Hard mining on normalized embeddings.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        if len(embeddings) < 3:
            return embeddings.new_tensor(0.0, requires_grad=True)
            
        # Ensure L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute squared Euclidean distances: ||z_i - z_j||^2 = 2 - 2 * cos_sim
        dot_product = torch.matmul(embeddings, embeddings.t())
        distances = 2.0 - 2.0 * dot_product
        distances = torch.clamp(distances, min=0.0)
        
        # Mask for positive pairs (same class, different anchors)
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        mask_pos = label_equal & ~indices_equal
        
        # Mask for negative pairs (different classes)
        mask_neg = ~label_equal
        
        triplet_loss = []
        N = len(embeddings)
        for i in range(N):
            pos_indices = torch.where(mask_pos[i])[0]
            neg_indices = torch.where(mask_neg[i])[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            # Find the hardest positive (maximum distance)
            hard_pos = distances[i, pos_indices].max()
            # Find the hardest negative (minimum distance)
            hard_neg = distances[i, neg_indices].min()
            
            loss_temp = hard_pos - hard_neg + self.margin
            triplet_loss.append(torch.clamp(loss_temp, min=0.0))
            
        if len(triplet_loss) == 0:
            return embeddings.new_tensor(0.0, requires_grad=True)
            
        return torch.stack(triplet_loss).mean()


class DwoPPLoss(nn.Module):
    """
    Distillation without Positive Pairs (L_DwoPP).
    Distills knowledge from the old model by comparing probability distributions
    over previous classes EXCLUDING the positive/ground-truth class.
    """
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, new_embeddings, old_embeddings, labels, class_embeddings, prev_intro_cls):
        if len(new_embeddings) == 0 or prev_intro_cls <= 0:
            return new_embeddings.new_tensor(0.0, requires_grad=True)
            
        # Ensure normalization
        new_embeddings = F.normalize(new_embeddings, p=2, dim=1)
        old_embeddings = F.normalize(old_embeddings, p=2, dim=1)
        class_embeddings = F.normalize(class_embeddings, p=2, dim=1)
        
        # Cosine similarity logits: (N, C_prev)
        logits_new = torch.matmul(new_embeddings, class_embeddings.t()) / self.temperature
        logits_old = torch.matmul(old_embeddings, class_embeddings.t()) / self.temperature
        
        kl_losses = []
        for i in range(len(new_embeddings)):
            label = labels[i].item()
            
            # Identify valid classes for distillation (previous classes excluding target label)
            valid_indices = [c for c in range(prev_intro_cls) if c != label]
            if len(valid_indices) == 0:
                continue
            valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=new_embeddings.device)
            
            # Softmax over valid (non-positive) class prototypes
            p_new = F.softmax(logits_new[i, valid_indices_tensor], dim=0)
            p_old = F.softmax(logits_old[i, valid_indices_tensor], dim=0)
            
            # KL Divergence: KL(p_old || p_new) = sum( p_old * log(p_old / p_new) )
            kl = torch.sum(p_old * torch.log((p_old + 1e-12) / (p_new + 1e-12)))
            kl_losses.append(kl)
            
        if len(kl_losses) == 0:
            return new_embeddings.new_tensor(0.0, requires_grad=True)
            
        return torch.stack(kl_losses).mean()


class AssignerWrapper(nn.Module):
    """
    Wrapper to intercept and store the latest ground truth assignment results
    to link positive anchors with correct classes in the loss computation.
    """
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
    """
    Modified YOLO-World Head Module with a 1x1 256-dim Convolutional Projection Head
    to output normalized visual features for image retrieval.
    """
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
        
        # Project and normalized retrieval embeddings (L2 Normalized)
        ret_embed = ret_pred(img_feat)
        ret_embed = F.normalize(ret_embed, p=2, dim=1)
        
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
    """
    Lifelong Image Retrieval Head integrating Triplet Episodic Loss and DwoPP Distillation Loss.
    """
    def __init__(self, 
                 *args, 
                 loss_retrieval_weight=0.5, 
                 loss_dwopp_weight=0.5,
                 retrieval_dim=256, 
                 triplet_margin=0.3, 
                 dwopp_temperature=0.05,
                 text_channels=512,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_retrieval_weight = loss_retrieval_weight
        self.loss_dwopp_weight = loss_dwopp_weight
        self.retrieval_dim = retrieval_dim
        self.triplet_loss = EpisodicHardMiningLoss(margin=triplet_margin)
        self.dwopp_loss = DwoPPLoss(temperature=dwopp_temperature)
        
        # Projection layer mapping 512-dim text features into 256-dim retrieval space
        self.text_projection = nn.Linear(text_channels, retrieval_dim)
        
        # Frozen teacher model for distillation, initialized dynamically during task > 1
        self.old_head_module = None
        
        if hasattr(self, 'assigner') and self.assigner is not None:
            self.assigner = AssignerWrapper(self.assigner)
            
    def init_old_model_if_needed(self):
        """Copies and freezes current loaded weights as the old model (theta_t-1) for DwoPP."""
        if self.prev_intro_cls > 0 and self.old_head_module is None:
            print(f"[OurHeadRetrieval] Creating frozen teacher copy from current state for DwoPP distillation.")
            self.old_head_module = copy.deepcopy(self.head_module)
            for p in self.old_head_module.parameters():
                p.requires_grad = False
            self.old_head_module.eval()

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict], fusion_att: bool=False) -> dict:
        
        if hasattr(self, 'assigner') and self.assigner is not None:
            if not isinstance(self.assigner, AssignerWrapper):
                self.assigner = AssignerWrapper(self.assigner)
                
        # Forward pass of retrieval head
        outs = self(img_feats, txt_feats)
        cls_scores, bbox_preds, bbox_dist_preds, ret_embeds = outs
        det_outs = (cls_scores, bbox_preds, bbox_dist_preds)
        
        # 1. Detection losses
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
            
        # 2. Metric learning and Distillation Losses
        if not hasattr(self, 'assigner') or self.assigner is None or self.assigner.latest_result is None:
            losses['loss_retrieval'] = ret_embeds[0].new_tensor(0.0, requires_grad=True)
            losses['loss_dwopp'] = ret_embeds[0].new_tensor(0.0, requires_grad=True)
            return losses
            
        assign_result = self.assigner.latest_result
        
        # Flatten and concatenate multi-level retrieval features: (batch_size, total_anchors, retrieval_dim)
        flatten_ret_embeds = []
        for ret_embed in ret_embeds:
            b, c, h, w = ret_embed.shape
            flat = ret_embed.view(b, c, -1).permute(0, 2, 1)
            flatten_ret_embeds.append(flat)
        flatten_ret_embeds = torch.cat(flatten_ret_embeds, dim=1)
        
        pos_embeddings_list = []
        pos_labels_list = []
        
        # Extract matched positive anchors and labels
        if isinstance(assign_result, dict):
            fg_mask_pre_prior = assign_result['fg_mask_pre_prior']
            assigned_bboxes = assign_result['assigned_bboxes']
            
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
                dists = torch.abs(img_pos_assigned_boxes.unsqueeze(1) - img_gt_boxes.unsqueeze(0)).sum(dim=-1)
                pos_gt_indices = torch.argmin(dists, dim=-1)
                img_pos_labels = img_gt_labels[pos_gt_indices]
                
                pos_embeddings_list.append(img_pos_embeds)
                pos_labels_list.append(img_pos_labels)
        else:
            if isinstance(assign_result, list):
                gt_inds_list = [res.gt_inds for res in assign_result]
            else:
                gt_inds_list = assign_result.gt_inds
                if len(gt_inds_list.shape) == 2:
                    gt_inds_list = [gt_inds_list[i] for i in range(gt_inds_list.shape[0])]
                else:
                    gt_inds_list = [gt_inds_list]
                    
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
                
        # Episodic Loss (L_eps)
        if len(pos_embeddings_list) > 0:
            all_pos_embeds = torch.cat(pos_embeddings_list, dim=0)
            all_pos_labels = torch.cat(pos_labels_list, dim=0)
            loss_eps = self.triplet_loss(all_pos_embeds, all_pos_labels)
        else:
            all_pos_embeds = None
            loss_eps = ret_embeds[0].new_tensor(0.0, requires_grad=True)
            
        losses['loss_retrieval'] = loss_eps * self.loss_retrieval_weight
        
        # DwoPP Distillation Loss (L_DwoPP)
        loss_dwopp = ret_embeds[0].new_tensor(0.0, requires_grad=True)
        if self.prev_intro_cls > 0 and all_pos_embeds is not None:
            self.init_old_model_if_needed()
            
            # Forward pass through frozen old head module
            with torch.no_grad():
                old_outs = self.old_head_module(img_feats, txt_feats)
                old_ret_embeds = old_outs[3]
                
            # Flatten old model features
            flatten_old_ret_embeds = []
            for old_ret_embed in old_ret_embeds:
                b, c, h, w = old_ret_embed.shape
                flat = old_ret_embed.view(b, c, -1).permute(0, 2, 1)
                flatten_old_ret_embeds.append(flat)
            flatten_old_ret_embeds = torch.cat(flatten_old_ret_embeds, dim=1)
            
            # Gather old embeddings for the same positive anchors
            old_pos_embeddings_list = []
            if isinstance(assign_result, dict):
                for i in range(fg_mask_pre_prior.shape[0]):
                    fg_mask = fg_mask_pre_prior[i]
                    if fg_mask.any():
                        old_pos_embeddings_list.append(flatten_old_ret_embeds[i][fg_mask])
            else:
                for i, gt_inds in enumerate(gt_inds_list):
                    fg_mask = gt_inds > 0
                    if fg_mask.any():
                        old_pos_embeddings_list.append(flatten_old_ret_embeds[i][fg_mask])
                        
            if len(old_pos_embeddings_list) > 0:
                all_old_pos_embeds = torch.cat(old_pos_embeddings_list, dim=0)
                
                # Fetch text embeddings of the previous classes
                prev_class_embeddings = txt_feats[0, :self.prev_intro_cls, :]
                
                # Project text prototypes into retrieval space
                class_prototypes = self.text_projection(prev_class_embeddings)
                
                # Compute distillation without positive pairs
                loss_dwopp = self.dwopp_loss(
                    all_pos_embeds,
                    all_old_pos_embeds,
                    all_pos_labels,
                    class_prototypes,
                    self.prev_intro_cls
                )
                
        losses['loss_dwopp'] = loss_dwopp * self.loss_dwopp_weight
        return losses

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False, 
                fusion_att: bool = False) -> InstanceList:
        
        # Forward pass returning 3 elements in eval mode
        x = self(img_feats, txt_feats)
        cls_scores, bbox_preds, ret_embeds = x
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
            
        # Align visual features of detected boundary boxes using ROI Align pooling
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
                feat = ret_embeds[level_idx][i:i+1] # (1, retrieval_dim, h, w)
                lvl_boxes = scaled_bboxes / stride
                rois = torch.cat([lvl_boxes.new_zeros(len(lvl_boxes), 1), lvl_boxes], dim=1)
                pooled = tv_ops.roi_align(feat, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
                pooled_feats.append(pooled.view(len(lvl_boxes), -1))
                
            # Compute average embedding across level feature maps
            img_box_embeds = torch.stack(pooled_feats, dim=0).mean(dim=0)
            img_box_embeds = F.normalize(img_box_embeds, p=2, dim=1)
            pred.features = img_box_embeds
            
        return predictions

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Ignore missing text_projection and old_head_module key warnings when loading detection checkpoints
        key_weight = prefix + 'text_projection.weight'
        key_bias = prefix + 'text_projection.bias'
        if key_weight not in state_dict:
            print(f"[OurHeadRetrieval] {key_weight} not found in state_dict. Initializing text_projection randomly.")
            if key_weight in missing_keys:
                missing_keys.remove(key_weight)
            if key_bias in missing_keys:
                missing_keys.remove(key_bias)
                
        for key in list(missing_keys):
            if 'old_head_module' in key:
                missing_keys.remove(key)
        for key in list(unexpected_keys):
            if 'old_head_module' in key:
                unexpected_keys.remove(key)
                
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
