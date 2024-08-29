# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/bravodataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
'''
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=['leftImg8bit/train', 'leftImg8bit/train_extra', 'leftImg8bit/val'],
        ann_dir=['gtFine/train', 'gtCoarse/train_extra', 'gtFine/val'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=['leftImg8bit/train', 'leftImg8bit/train_extra', 'leftImg8bit/val'],
        ann_dir=['gtFine/train', 'gtCoarse/train_extra', 'gtFine/val'],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=['leftImg8bit/train', 'leftImg8bit/train_extra', 'leftImg8bit/val'],
        ann_dir=['gtFine/train', 'gtCoarse/train_extra', 'gtFine/val'],
        pipeline=test_pipeline))
'''
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=['bravo_ACDC', 'bravo_outofcontext', 'bravo_SMIYC', 'bravo_synobjs', 'bravo_synflare', 'bravo_synrain'],
        ann_dir=['bravo_ACDC', 'bravo_outofcontext', 'bravo_SMIYC', 'bravo_synobjs', 'bravo_synflare', 'bravo_synrain'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=['bravo_ACDC/fog/test', 'bravo_ACDC/night/test', 'bravo_ACDC/rain/test', 'bravo_ACDC/snow/test', 'bravo_outofcontext', 'bravo_SMIYC', 'bravo_synobjs', 'bravo_synflare', 'bravo_synrain'],
        ann_dir=['bravo_ACDC/fog/test', 'bravo_ACDC/night/test', 'bravo_ACDC/rain/test', 'bravo_ACDC/snow/test', 'bravo_outofcontext', 'bravo_SMIYC', 'bravo_synobjs', 'bravo_synflare', 'bravo_synrain'],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=[
            'bravo_synobjs/armchair', 'bravo_synobjs/bathtub', 'bravo_synobjs/billboard', 'bravo_synobjs/cheetah', 'bravo_synobjs/elephant',
            'bravo_synobjs/baby', 'bravo_synobjs/bench', 'bravo_synobjs/box', 'bravo_synobjs/chimpanzee', 'bravo_synobjs/flamingo',
            'bravo_synobjs/giraffe', 'bravo_synobjs/hippopotamus', 'bravo_synobjs/koala', 'bravo_synobjs/lion', 'bravo_synobjs/panda',
            'bravo_synobjs/gorilla', 'bravo_synobjs/kangaroo', 'bravo_synobjs/penguin', 'bravo_synobjs/plant', 'bravo_synobjs/polar bear',
            'bravo_synobjs/sofa', 'bravo_synobjs/tiger', 'bravo_synobjs/vase', 'bravo_synobjs/table', 'bravo_synobjs/toilet',
            'bravo_synobjs/zebra','bravo_ACDC/fog/test', 'bravo_ACDC/night/test', 'bravo_ACDC/rain/test', 'bravo_ACDC/snow/test', 'bravo_outofcontext', 'bravo_SMIYC', 'bravo_synflare', 'bravo_synrain'],
        ann_dir=[
            'bravo_synobjs/armchair', 'bravo_synobjs/bathtub', 'bravo_synobjs/billboard', 'bravo_synobjs/cheetah', 'bravo_synobjs/elephant',
            'bravo_synobjs/baby', 'bravo_synobjs/bench', 'bravo_synobjs/box', 'bravo_synobjs/chimpanzee', 'bravo_synobjs/flamingo',
            'bravo_synobjs/giraffe', 'bravo_synobjs/hippopotamus', 'bravo_synobjs/koala', 'bravo_synobjs/lion', 'bravo_synobjs/panda',
            'bravo_synobjs/gorilla', 'bravo_synobjs/kangaroo', 'bravo_synobjs/penguin', 'bravo_synobjs/plant', 'bravo_synobjs/polar bear',
            'bravo_synobjs/sofa', 'bravo_synobjs/tiger', 'bravo_synobjs/vase', 'bravo_synobjs/table', 'bravo_synobjs/toilet',
            'bravo_synobjs/zebra','bravo_ACDC/fog/test', 'bravo_ACDC/night/test', 'bravo_ACDC/rain/test', 'bravo_ACDC/snow/test', 'bravo_outofcontext', 'bravo_SMIYC', 'bravo_synflare', 'bravo_synrain'],
        pipeline=test_pipeline))
