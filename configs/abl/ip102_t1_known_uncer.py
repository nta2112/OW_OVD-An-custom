_base_ = './ip102_t1_sim_restr.py'

model = dict(
    bbox_head=dict(
        use_known_uncertainty=True
    )
)