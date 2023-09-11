_base_ = './fpn_simple_vit_adapter_large_448_320k_concat_cos_dist_vl.py'

model = dict(decode_head=dict(type='SimpleHeadRelSemVL',
                              loss_decode=dict(
                                  type='RelativeSemanticLoss',
                                  dataset='concat_cityscapes_split',
                                  txt2idx_star_path='./txt2idx_star_exp03.pkl',
                                  idx_star2emb_path='./idx_star2emb_exp03.pkl',
                              )))
