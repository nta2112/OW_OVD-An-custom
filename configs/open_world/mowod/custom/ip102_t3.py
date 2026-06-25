_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# Suppress MMEngine's extremely verbose configuration printing at startup
import mmengine
try:
    mmengine.Config.pretty_text = property(lambda self: "[Config dump suppressed for cleaner logs]")
    # Suppress verbose model architecture logging
    from mmengine.logging import MMLogger
    _orig_info = MMLogger.info
    def _clean_info(self, msg, *args, **kwargs):
        msg_str = str(msg)
        if 'OurDetector(' in msg_str or 'MultiModalYOLOBackbone(' in msg_str or 'paramwise_options' in msg_str:
            return
        _orig_info(self, msg, *args, **kwargs)
    MMLogger.info = _clean_info
except Exception:
    pass

# Fool-proof monkey patch to fix double 'test/test/' path bug in Kaggle datasets
try:
    from mmyolo.datasets import YOLOv5CocoDataset
    _orig_parse = YOLOv5CocoDataset.parse_data_info
    def _patched_parse(self, raw_data_info):
        data_info = _orig_parse(self, raw_data_info)
        if 'img_path' in data_info and 'test/test/' in data_info['img_path']:
            data_info['img_path'] = data_info['img_path'].replace('test/test/', 'test/')
        return data_info
    YOLOv5CocoDataset.parse_data_info = _patched_parse
except Exception:
    pass

# Dynamically load class names from IP102 annotations on Kaggle
import json
try:
    with open('/kaggle/input/datasets/eljazouly/ip102-coco-annotations/coco_annotations/train.json', 'r') as f:
        coco_data = json.load(f)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    del f, coco_data, categories
except Exception:
    # Fallback placeholder list for local validation
    class_names = [f"pest_{i}" for i in range(102)]
finally:
    try:
        del json
    except NameError:
        pass

# open world setting
prev_intro_cls = 6
cur_intro_cls = 3
train_json = '/kaggle/input/datasets/eljazouly/ip102-coco-annotations/coco_annotations/train.json'
test_json = '/kaggle/input/datasets/eljazouly/ip102-coco-annotations/coco_annotations/test.json'
embedding_path = 'data/IP102/ip102_gt_embeddings.npy'
att_embeddings = 'data/IP102/task_att_1_embeddings.pth'
pipline = [dict(type='att_select', log_start_epoch=1)]
thr = 0.55
alpha = 0.2
use_sigmoid = True
distributions = 'data/IP102/mowod_distribution_sim1.pth'
top_k = 10

# yolo world setting
num_classes = 9
num_training_classes = prev_intro_cls + cur_intro_cls  # 9
max_epochs = 20
close_mosaic_epochs = 5
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 1e-4
weight_decay = 0.05
train_batch_size_per_gpu = 24
load_from = 'pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth'
persistent_workers = True

# model settings
model = dict(type='OurDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path=embedding_path,
             prompt_dim=text_channels,
             num_prompts=len(class_names),
             pipline=pipline,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           frozen_stages=4,
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=False,              # Unfreeze PAFPN neck to learn pest features
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             bbox_head=dict(type='OurHead',
                            att_embeddings=att_embeddings,
                            thr=thr,
                            alpha=alpha,
                            use_sigmoid=use_sigmoid,
                            distributions=distributions,
                            prev_intro_cls=prev_intro_cls,
                            cur_intro_cls=cur_intro_cls,
                            top_k=top_k,
                            head_module=dict(
                                type='OurHeadModule',
                                freeze_all=False,     # Unfreeze Head module to align with text embeddings
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes,),),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))


# dataset settings
coco_train_dataset = dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=class_names[:9]),  # Learn only the first 9 classes (t1 + t2 + t3)
            data_root='/kaggle/input/datasets/rtlmhjbn/ip02-dataset/classification/',
            ann_file=train_json,
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32)),
        class_text_path='data/texts/IP102/class_texts.json',
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

custom_hooks = [
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2),
    dict(type='OurWorkPiplineHook')
]

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals, max_keep_ckpts=2, save_best='Current class AP50', rule='greater',
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=max_epochs,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=1,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                      _base_.val_interval_stage2)])

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(bias_decay_mult=0.0,
                       norm_decay_mult=0.0,
                       custom_keys={
                           'backbone.text_model':
                           dict(lr_mult=0.01),
                           'logit_scale':
                           dict(weight_decay=0.0),
                           'embeddings':
                           dict(weight_decay=0.0)
                       }),
    constructor='YOLOWv5OptimizerConstructor')

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]

test_dataloader = dict(batch_size=24,
                        dataset=dict(type='YOLOv5CocoDataset',
                        metainfo=dict(classes=class_names[:9]),  # Evaluate on first 9 classes
                        data_root='/kaggle/input/datasets/rtlmhjbn/ip02-dataset/classification/',
                        ann_file=test_json,
                        data_prefix=dict(img=''),
                        filter_cfg=dict(filter_empty_gt=True, min_size=32),
                        pipeline=test_pipeline)
                       )

test_evaluator = dict(_delete_=True,
                      type='OWODEvaluator',
                      cfg=dict(
                         dataset_root='data/IP102/voc/',
                         file_name='mowod/all_task_test.txt',
                         prev_intro_cls=prev_intro_cls,
                         cur_intro_cls=cur_intro_cls,
                         unknown_id=102,
                         class_names=class_names[:9]
                      )
                     )
val_evaluator = test_evaluator
val_dataloader = test_dataloader
find_unused_parameters = True
