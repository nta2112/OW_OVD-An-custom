_base_ = '../open_world/mowod/custom/ip102_t1.py'

train_batch_size_per_gpu = 24

model = dict(
    bbox_head=dict(
        select_all_attr=True
    )
)
