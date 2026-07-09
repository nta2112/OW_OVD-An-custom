_base_ = './ip102_t1_sim_restr.py'

# Evaluation should rebuild the same 675-attribute head saved in the checkpoint.
# The training config starts from the full attribute bank and writes this file
# after attribute selection.
att_embeddings = 'data/IP102/selected_att_embeddings_sim_restr.pth'
pipline = []

model = dict(
    pipline=pipline,
    bbox_head=dict(att_embeddings=att_embeddings)
)
