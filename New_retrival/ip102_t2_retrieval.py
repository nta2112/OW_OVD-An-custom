_base_ = '../configs/open_world/mowod/custom/ip102_t2.py'

# Dynamically import our custom head from the local folder
custom_imports = dict(
    imports=['New_retrival.our_head_retrieval'],
    allow_failed_imports=False
)

# Replace box head types to use our new retrieval head and module
model = dict(
    bbox_head=dict(
        type='OurHeadRetrieval',
        retrieval_dim=256,
        loss_retrieval_weight=0.5,
        triplet_margin=0.3,
        head_module=dict(
            type='OurHeadRetrievalModule',
            retrieval_dim=256
        )
    )
)

# Training configurations: 1 epoch, batch size 16
max_epochs = 1
close_mosaic_epochs = 1
train_batch_size_per_gpu = 16

train_cfg = dict(max_epochs=max_epochs, val_interval=999)
train_dataloader = dict(
    num_workers=2,
    persistent_workers=False
)
