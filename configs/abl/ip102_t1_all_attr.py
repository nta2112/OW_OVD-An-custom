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

custom_hooks = []
for hook in _base_.custom_hooks:
    if hook.get('type') == 'EarlyStoppingHook':
        new_hook = hook.copy()
        new_hook['patience'] = 1
        new_hook['min_delta'] = 5.0
        custom_hooks.append(new_hook)
    else:
        custom_hooks.append(hook)
