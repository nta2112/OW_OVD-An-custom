_base_ = './ip102_t1_all_attr_ood.py'

model = dict(
    bbox_head=dict(
        select_all_attr=False,
        selected_att_path='data/IP102/selected_att_embeddings.pth',
        attr_sel_for_known_only=False,
        use_top_k_att=False,
        use_ood_gate=True,
        use_ood_prob=False,
        use_known_uncertainty=False
    )
)

custom_hooks = [hook for hook in _base_.custom_hooks if hook.get('type') != 'EarlyStoppingHook']