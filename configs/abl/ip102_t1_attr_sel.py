_base_ = './ip102_t1_all_attr.py'

model = dict(
    bbox_head=dict(
        select_all_attr=False,
        selected_att_path='data/IP102/selected_att_embeddings.pth'
    )
)
