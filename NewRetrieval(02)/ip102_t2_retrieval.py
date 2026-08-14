_base_ = '../configs/test/ip102_t2.py'

custom_imports = dict(
    imports=['NewRetrieval(02).our_head_retrieval_dwopp'],
    allow_failed_imports=False
)

model = dict(
    bbox_head=dict(
        type='OurHeadRetrieval',
        retrieval_dim=256,
        loss_retrieval_weight=0.5,
        loss_dwopp_weight=0.5,
        triplet_margin=0.3,
        dwopp_temperature=0.05,
        text_channels=512,
        head_module=dict(
            type='OurHeadRetrievalModule',
            retrieval_dim=256
        )
    )
)

max_epochs = 1
close_mosaic_epochs = 1
train_batch_size_per_gpu = 16

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
