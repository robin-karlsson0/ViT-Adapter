_base_ = [
    '../_base_/datasets/concat_coco_stuff164k_orig_beit.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]
crop_size = (896, 896)
pretrained = 'pretrained/beit_large_patch16_224_pt22k_ft22k.pth'
model = dict(
    type='EncoderDecoderVL',
    pretrained=pretrained,
    backbone=dict(
        type='BEiTAdapter',
        img_size=896,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        freeze_backbone=False,
    ),
    neck=dict(type='FPNVL',
              in_channels=[1024, 1024, 1024, 1024],
              out_channels=1024,
              num_outs=4,
              norm_cfg=dict(type='SyncBN', requires_grad=True),
              use_last_feat_map_only=True),
    decode_head=dict(type='SimpleHeadVL',
                     in_channels=[1024, 1024, 1024, 1024],
                     channels=768,
                     output_size=(896, 896),
                     normalize_output=True,
                     normalize_target_embs=True,
                     norm_cfg=dict(type='SyncBN', requires_grad=True),
                     align_corners=False,
                     idx_star2emb_path='idx_star2emb_cseg_coco_orig.pkl',
                     loss_decode=dict(type='CosineEmbMaskLoss',
                                      margin=0.5,
                                      loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(896, 896)))
# dataset settings
# BEiT img encoder
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1000000000000, 896), ratio_range=(0.5, 1)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0,
         seg_pad_val=4294967295),  # np.iinfo(np.uint32).max
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-4,  # 2.625e-4,  # 6e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))
lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=0.0,
                 by_epoch=False)
data = dict(samples_per_gpu=1, train=dict(pipeline=train_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=100)
runner = dict(type='IterBasedRunner')
evaluation = dict(interval=32000,
                  metric='mIoU',
                  save_best='mIoU',
                  efficient_test=True,
                  thresh_smpls=1000)  # Per GPU = thresh_smapls / #GPU ?
