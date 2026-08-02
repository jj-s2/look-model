ann_file_test = 'C:/Users/John/Desktop/look model/experiments/outputs/losocv/fold_S4/test.pkl'
ann_file_train = 'C:/Users/John/Desktop/look model/experiments/outputs/losocv/fold_S4/train.pkl'
ann_file_val = 'C:/Users/John/Desktop/look model/experiments/outputs/losocv/fold_S4/val.pkl'
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
left_kp = [
    1,
    3,
    5,
    7,
    9,
    11,
    13,
    15,
]
load_from = 'C:\\Users\\John\\Desktop\\look model\\experiments\\outputs\\losocv\\fold_S4\\best_acc_top1_epoch_10.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        base_channels=32,
        conv1_stride_s=1,
        depth=50,
        dilations=(
            1,
            1,
            1,
        ),
        in_channels=17,
        inflate=(
            0,
            1,
            1,
        ),
        num_stages=3,
        out_indices=(2, ),
        pool1_stride_s=1,
        pretrained=None,
        spatial_strides=(
            2,
            2,
            2,
        ),
        stage_blocks=(
            4,
            6,
            3,
        ),
        temporal_strides=(
            1,
            1,
            2,
        ),
        type='ResNet3dSlowOnly'),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.7,
        in_channels=512,
        num_classes=2,
        type='I3DHead'),
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.001))
param_scheduler = [
    dict(
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
right_kp = [
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'C:/Users/John/Desktop/look model/experiments/outputs/losocv/fold_S4/test.pkl',
        pipeline=[
            dict(
                clip_len=48,
                num_clips=10,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
            dict(scale=(
                -1,
                64,
            ), type='Resize'),
            dict(crop_size=64, type='CenterCrop'),
            dict(
                double=True,
                left_kp=[
                    1,
                    3,
                    5,
                    7,
                    9,
                    11,
                    13,
                    15,
                ],
                right_kp=[
                    2,
                    4,
                    6,
                    8,
                    10,
                    12,
                    14,
                    16,
                ],
                sigma=0.6,
                type='GeneratePoseTarget',
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(input_format='NCTHW_Heatmap', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        split=None,
        test_mode=True,
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
    dict(
        out_file_path=
        'C:\\Users\\John\\Desktop\\look model\\experiments\\outputs\\losocv\\fold_S4\\test_results.pkl',
        type='DumpResults'),
]
test_pipeline = [
    dict(
        clip_len=48, num_clips=10, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
    dict(scale=(
        -1,
        64,
    ), type='Resize'),
    dict(crop_size=64, type='CenterCrop'),
    dict(
        double=True,
        left_kp=[
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
        ],
        right_kp=[
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
        ],
        sigma=0.6,
        type='GeneratePoseTarget',
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(input_format='NCTHW_Heatmap', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=5)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        'C:/Users/John/Desktop/look model/experiments/outputs/losocv/fold_S4/train.pkl',
        pipeline=[
            dict(clip_len=48, type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
            dict(scale=(
                -1,
                64,
            ), type='Resize'),
            dict(area_range=(
                0.56,
                1.0,
            ), type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                56,
                56,
            ), type='Resize'),
            dict(
                flip_ratio=0.5,
                left_kp=[
                    1,
                    3,
                    5,
                    7,
                    9,
                    11,
                    13,
                    15,
                ],
                right_kp=[
                    2,
                    4,
                    6,
                    8,
                    10,
                    12,
                    14,
                    16,
                ],
                type='Flip'),
            dict(
                sigma=0.6,
                type='GeneratePoseTarget',
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(input_format='NCTHW_Heatmap', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        split=None,
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=48, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
    dict(scale=(
        -1,
        64,
    ), type='Resize'),
    dict(area_range=(
        0.56,
        1.0,
    ), type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        56,
        56,
    ), type='Resize'),
    dict(
        flip_ratio=0.5,
        left_kp=[
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
        ],
        right_kp=[
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
        ],
        type='Flip'),
    dict(
        sigma=0.6,
        type='GeneratePoseTarget',
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(input_format='NCTHW_Heatmap', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        'C:/Users/John/Desktop/look model/experiments/outputs/losocv/fold_S4/val.pkl',
        pipeline=[
            dict(
                clip_len=48,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
            dict(scale=(
                -1,
                64,
            ), type='Resize'),
            dict(crop_size=64, type='CenterCrop'),
            dict(
                sigma=0.6,
                type='GeneratePoseTarget',
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(input_format='NCTHW_Heatmap', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        split=None,
        test_mode=True,
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(clip_len=48, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
    dict(scale=(
        -1,
        64,
    ), type='Resize'),
    dict(crop_size=64, type='CenterCrop'),
    dict(
        sigma=0.6,
        type='GeneratePoseTarget',
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(input_format='NCTHW_Heatmap', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'C:\\Users\\John\\Desktop\\look model\\experiments\\outputs\\losocv\\fold_S4'
