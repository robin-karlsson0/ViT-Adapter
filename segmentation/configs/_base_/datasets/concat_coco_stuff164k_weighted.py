# dataset settings
dataset_type = 'CompSemCOCOCsegDataset'
data_root = 'data/concat_coco_cseg_weighted/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# CLIP img encoder
img_norm_cfg = dict(mean=[122.7709383, 116.7460125, 104.09373615],
                    std=[68.5005327, 66.6321579, 70.32316305],
                    to_rgb=True)
crop_size = (448, 448)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(100000000000, 448),
         ratio_range=(0.5, 2.0)),  # 896
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0,
         seg_pad_val=4294967295),  # np.iinfo(np.uint32).max
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(100000000000, 448),  # 896
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(samples_per_gpu=4,
            workers_per_gpu=4,
            train=dict(type=dataset_type,
                       data_root=data_root,
                       img_dir='imgs/train',
                       ann_dir='anns/train',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     data_root=data_root,
                     img_dir='imgs/val_500',
                     ann_dir='anns/val_500',
                     pipeline=test_pipeline),
            test=dict(type=dataset_type,
                      data_root=data_root,
                      img_dir='imgs/val',
                      ann_dir='anns/val',
                      pipeline=test_pipeline))