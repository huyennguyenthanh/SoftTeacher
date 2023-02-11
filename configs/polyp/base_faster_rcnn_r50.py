_base_ = "/home/nguyen.thanh.huyenb/yolov7/SoftTeacher/configs/baseline/base.py"
fold = 1
percent = 5
classes = ('polyp',)
num_classes=1
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        ann_file="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/Semi_PolypsSet/semi_1_labels/train2019.${fold}@${percent}.json",
        img_prefix="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/PolypsSet/train2019/images/",
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
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
# lr_config = dict(step=[16, 22])
# runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=30)
# checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=10)
# evaluation = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="polyp_soft_teacher",
                name="${cfg_name}_percent_${percent}_fold_${fold}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
