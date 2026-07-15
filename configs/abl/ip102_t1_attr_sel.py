_base_ = './ip102_t1_all_attr_ood.py'

model = dict(
    bbox_head=dict(
        select_all_attr=False,
        selected_att_path='data/IP102/selected_att_embeddings.pth',
        attr_sel_for_known_only=False,
        use_top_k_att=False,
        use_ood_gate=True
    )
)

custom_hooks = [
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=0,
         switch_pipeline=_base_.train_pipeline_stage2),
    dict(type='OurWorkPiplineHook'),
    dict(type='EarlyStoppingHook',
         monitor='coco/Current class AP50',
         rule='greater',
         patience=2,
         min_delta=10.0)
]