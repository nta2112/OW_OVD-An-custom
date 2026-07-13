# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOWorldGTDetector(YOLODetector):
    """Implementation of YOLO World Baseline + Ground Truth Detector"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if len(embedding_path) > 0:
            import numpy as np
            self.embeddings = torch.nn.Parameter(
                torch.from_numpy(np.load(embedding_path)).float())
        else:
            # random init
            embeddings = nn.functional.normalize(torch.randn(
                (num_prompts, prompt_dim)),
                                                 dim=-1)
            self.embeddings = nn.Parameter(embeddings)

        if self.freeze_prompt:
            self.embeddings.requires_grad = False
        else:
            self.embeddings.requires_grad = True

        if use_mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                nn.Linear(prompt_dim * 2, prompt_dim))
        else:
            self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # 1. Compute raw logits for all 102 classes
        self.bbox_head.num_classes = self.num_test_classes
        outs = self.bbox_head(img_feats, txt_feats)
        
        # 2. Extract known class logits and compute max over unknown class logits
        cls_logits = outs[0]
        known_logits = [c[:, :self.num_training_classes, :, :] for c in cls_logits]
        unknown_logits = [c[:, self.num_training_classes:, :, :] for c in cls_logits]
        max_unknown_logits = [u.max(dim=1, keepdim=True)[0] for u in unknown_logits]
        
        # 3. Concat known class logits with the max unknown logit
        new_cls_logits = [torch.cat([k, m], dim=1) for k, m in zip(known_logits, max_unknown_logits)]
        new_outs = (new_cls_logits, *outs[1:])
        
        # 4. Run post-processing on 28 classes (27 known + 1 unknown)
        self.bbox_head.num_classes = self.num_training_classes + 1
        
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        results_list = self.bbox_head.predict_by_feat(*new_outs,
                                                      batch_img_metas=batch_img_metas,
                                                      rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process."""
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        # use sliced embeddings based on train vs. test
        if self.training:
            txt_feats = self.embeddings[:self.num_training_classes][None]
        else:
            txt_feats = self.embeddings[:self.num_test_classes][None]

        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)

        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
