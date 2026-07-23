_base_ = '../custom/ip102_t1.py'

train_batch_size_per_gpu = 24

model = dict(
    bbox_head=dict(
        select_all_attr=True,
        use_top_k_att=False,
        use_ood_gate=False,
        use_known_uncertainty=False
    )
)
