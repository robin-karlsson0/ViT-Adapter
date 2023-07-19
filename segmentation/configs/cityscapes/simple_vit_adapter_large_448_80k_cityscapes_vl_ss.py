# Copyright (c) Shanghai AI Lab. All rights reserved.

# Does not work with default mmcv-full package
# import numpy as np

_base_ = [
    '../_base_/../_base_/datasets/cityscapes_448x448_vl.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# ''
crop_size = (448, 448)
# pretrained = 'https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth'
pretrained = 'pretrain/ViT-L_14_336px_clip_backbone.pth'
# load_from = 'https://github.com/czczup/ViT-Adapter/releases/download/v0.2.3/mask2former_beit_adapter_large_896_80k_mapillary.zip'
# load_from = 'pretrained/mask2former_beit_adapter_large_896_80k_mapillary.pth.tar'
# load_from = 'pretrain/ViT14_clip_backbone.pth'
model = dict(
    type='EncoderDecoderVL',
    pretrained=pretrained,
    backbone=dict(
        type='ViTAdapterVL',
        # ViT parameters
        img_size=(448, 448),
        patch_size=14,
        patch_bias=False,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.4,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        pre_norm=False,  # True
        final_norm=False,  # True
        return_qkv=True,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False,
        freeze_backbone=True,
        # ViT-Adapter parameters
        pretrain_size=448,  #336
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
    ),
    decode_head=dict(
        type='SimpleHeadVL',
        in_channels=[1024, 1024, 1024, 1024],
        channels=768,
        output_size=(448, 448),
        normalize_output=True,
        normalize_target_embs=True,
        norm_cfg=dict(type='SyncBN',
                      requires_grad=True),  # Or dict(type='LN', eps=1e-6), ?
        align_corners=False,
        loss_decode=dict(type='CosineEmbMaskLoss', margin=0.5,
                         loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(448, 448)))
# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(896, 448), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Pad',
        size=crop_size,
        pad_val=0,
        # seg_pad_val=np.iinfo(np.uint32).max),
        seg_pad_val=4294967295),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,  # 2e-5
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
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
runner = dict(type='IterBasedRunner')
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')
