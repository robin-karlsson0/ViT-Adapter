_base_ = './fpn_vit_adapter_large_448_320k_concat_cos_dist_vl.py'

model = dict(decode_head=dict(type='FPNHeadRelSemVL',
                              loss_decode=dict(
                                  type='RelativeSemanticLoss',
                                  dataset='concat',
                                  txt2idx_star_path='./txt2idx_star.pkl',
                                  idx_star2emb_path='./idx_star2emb.pkl',
                              )))
