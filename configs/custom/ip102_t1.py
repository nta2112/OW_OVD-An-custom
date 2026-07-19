_base_ = ('../../third_party/mmyolo/configs/yolov8/'
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
        if any(term in msg_str for term in ['OurDetector(', 'MultiModalYOLOBackbone(', 'paramwise_options', 'Checkpoints will be saved to']):
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

# Tự động tìm đường dẫn dataset trên Kaggle
import glob
import os
import json

dataset_root = None
for path in [
    '/kaggle/input/datasets/nta212/ip102-for-object-detection',
    '/kaggle/input/ip102-for-object-detection',
    'data/IP102',
    '.'
]:
    if os.path.exists(os.path.join(path, 'train.json')):
        dataset_root = path
        break

if dataset_root is None:
    paths = glob.glob('/kaggle/input/**/train.json', recursive=True)
    if paths:
        dataset_root = os.path.dirname(paths[0])

if dataset_root is None:
    dataset_root = '.'  # Fallback local path

train_json = os.path.join(dataset_root, 'train.json')
test_json = os.path.join(dataset_root, 'test.json')
val_json = os.path.join(dataset_root, 'val.json')
image_data_root = os.path.join(dataset_root, 'VOC2007/JPEGImages/')
if not os.path.exists(image_data_root):
    image_data_root = dataset_root

# Dynamically load class names from IP102 annotations
class_names = None
try:
    with open(train_json, 'r') as f:
        coco_data = json.load(f)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
except Exception:
    pass

if class_names is None:
    class_names = ['14', '15', '16', '18', '22', '23', '24', '25', '26', '37', '38', '39', '45', '46', '47', '48', '49', '50', '51', '66', '67', '69', '70', '86', '101']

# Clean up temporary variables to avoid deepcopy / pickle TypeError (cannot pickle 'TextIOWrapper' instances)
try:
    del json, os, glob, path, paths, f, coco_data, categories
except Exception:
    pass

# open world setting
prev_intro_cls = 0
cur_intro_cls = 7
embedding_path = 'data/IP102/ip102_gt_embeddings.npy'
att_embeddings = 'data/IP102/task_att_1_embeddings.pth'
pipline = [dict(type='att_select', log_start_epoch=1)]
thr = 0.55
alpha = 0.2
use_sigmoid = True
distributions = 'data/IP102/mowod_distribution_sim1.pth'
top_k = 10

# yolo world setting
num_classes = 7
num_training_classes = 7
max_epochs = 5
close_mosaic_epochs = max_epochs
save_epoch_intervals = 1
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
                       freeze_all=False,              # Frozen neck to run like original MOWOD and speed up
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
                                freeze_all=False,     # Frozen head module to run like original MOWOD and speed up
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
            metainfo=dict(classes=class_names[:7]),  # Learn the first 7 classes for T1
            data_root=image_data_root,
            ann_file=train_json,
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32)),
        class_text_path='data/texts/IP102/class_texts.json',
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        num_workers=4,                         # Parallel data loading to improve speed
                        pin_memory=True,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

custom_hooks = [
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2),
    dict(type='OurWorkPiplineHook'),
    dict(type='EarlyStoppingHook',                             # Early Stopping Hook configuration
         monitor='coco/Current class AP50',
         rule='greater',
         patience=2,                                           # Stop training if AP50 doesn't improve for 3 epochs
         min_delta=0.001)
]

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals, max_keep_ckpts=2, save_best='coco/Current class AP50', rule='greater',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),              # Less frequent logging (50 iterations) to clean up output
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=max_epochs,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=999,
                 dynamic_intervals=[(2, 1)])

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

test_dataloader = dict(
    _delete_=True,
    batch_size=24,
    num_workers=4,                         # Parallel data loading to improve speed
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='YOLOv5CocoDataset',
                 metainfo=dict(classes=class_names),  # Evaluate on all 25 classes
                 data_root=image_data_root,
                 ann_file=test_json,
                 data_prefix=dict(img=''),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=test_pipeline)
)

test_evaluator = dict(_delete_=True,
                      type='OWODEvaluator',
                      prefix='coco',                         # Explicitly define prefix for monitoring
                      cfg=dict(
                         dataset_root='data/IP102/voc/',
                         ann_file=test_json,
                         file_name='mowod/all_task_test.txt',
                         prev_intro_cls=prev_intro_cls,
                         cur_intro_cls=cur_intro_cls,
                         unknown_id=25,
                         class_names=class_names
                      )
                     )
val_dataloader = dict(
    _delete_=True,
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='YOLOv5CocoDataset',
                 metainfo=dict(classes=class_names),
                 data_root=image_data_root,
                 ann_file=val_json,
                 data_prefix=dict(img=''),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=test_pipeline)
)

val_evaluator = dict(_delete_=True,
                      type='OWODEvaluator',
                      prefix='coco',
                      cfg=dict(
                         dataset_root='data/IP102/voc_val/',
                         ann_file=val_json,
                         file_name='mowod/all_task_val.txt',
                         prev_intro_cls=prev_intro_cls,
                         cur_intro_cls=cur_intro_cls,
                         unknown_id=25,
                         class_names=class_names
                      )
                     )
find_unused_parameters = True
