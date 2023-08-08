_base_ = './fpn_simple_vit_adapter_large_448_160k_coco_cos_dist_vl.py'

model = dict(decode_head=dict(
    type='SimpleHeadRelSemVL',
    loss_decode=dict(
        type='RelativeSemanticLoss',
        dataset='coco',
        txt2idx_star_path='./txt2idx_star.pkl',
        idx_star2emb_path='./idx_star2emb.pkl',
    )))
