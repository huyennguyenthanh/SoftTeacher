# RUNTIME
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# SCHEDULE

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[16, 22])
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=50, val_interval=1)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=10)
evaluation = dict(interval=1)

# Models
model = dict(
    type='YOLOv3',
    backbone=dict(
        type='Darknet',
        depth=53
    ),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256
    ),
    bbox_head=dict(
        type='YOLOv3Head',
        num_classes=1,
        in_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=16,
            scales_per_octave=1,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=0.1111,
            loss_weight=2.0
        )
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms=dict(
            type='nms',
            iou_thr=0.5
        ),
        min_bbox_size=0,
        score_thr=0.05,
        max_per_img=100
    )
)



# Dataset 
fold = 1
percent = 10
classes = ('polyp',)
num_classes=1
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=5,
    train=dict(
        classes=classes,
        ann_file="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/Semi_PolypsSet/semi_1_labels/train2019.1@10.json",
        img_prefix="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/PolypsSet/valid/images/",
    ),
    val=dict(
        classes=classes,
        ann_file="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/Semi_PolypsSet/valid_1label.json",
        img_prefix="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/PolypsSet/valid/images/",
    ),
    test=dict(
        classes=classes,
        ann_file="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/Semi_PolypsSet/valid_1label.json",
        img_prefix="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/PolypsSet/valid/images/",
    ),
)
work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="polyp_yolov3_test",
                name="${cfg_name}_fold_{fold}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}"
                ),
            ),
            by_epoch=True,
        ),
    ],
)

# Distributed params 
dist_params = dict(backend='nccl')
