_base_ = './ip102_t1_attr_sel.py'

model = dict(
    bbox_head=dict(
        use_similarity_restriction=True,
        sim_restr_beta=0.2,
        selected_att_path='data/IP102/selected_att_embeddings_sim_restr.pth',
        use_known_uncertainty=False
    )
)
