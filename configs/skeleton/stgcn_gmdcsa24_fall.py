"""STGCN 微调配置：GMDCSA24 fall vs ADL 二分类。

用于与 PoseC3D 做骨干模型对比实验（LOSOCV）。
- 与 posec3d_slowonly_r50_gmdcsa24_fall.py 使用相同的 split pkl、相同的 epoch/lr/batch_size
- 输入格式不同：STGCN 用原始关键点序列（joint），PoseC3D 用热图
- graph_cfg layout='coco' 匹配 RTMPose 提取的 17 关键点
- load_from NTU60 预训练权重做迁移学习

运行命令（cwd=third_party/mmaction2）:
  python tools/train.py \
    ../../configs/skeleton/stgcn_gmdcsa24_fall.py \
    --work-dir ../../experiments/outputs/stgcn_gmdcsa24
"""
# ---- default_runtime 内联（避免 _base_ 相对路径依赖）----
default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
resume = False

# ---- 模型 ----
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=2, in_channels=256))

# ---- 数据集 ----
dataset_type = 'PoseDataset'
ann_file_train = 'datasets/annotations/gmdcsa24_pose_train.pkl'
ann_file_val = 'datasets/annotations/gmdcsa24_pose_val.pkl'
ann_file_test = 'datasets/annotations/gmdcsa24_pose_test.pkl'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        split=None,  # pkl 已是 list 格式
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        split=None,
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        split=None,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

# ---- 训练策略（与 PoseC3D 一致便于公平对比）----
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0003,
                   nesterov=True),
    clip_grad=dict(max_norm=40, norm_type=2))

# 从 NTU60 预训练权重迁移学习（cls_head 因 num_classes 不同会被跳过）
load_from = r'F:\look model\models\pretrained\stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.pth'
