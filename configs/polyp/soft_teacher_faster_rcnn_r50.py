_base_ = "/home/nguyen.thanh.huyenb/yolov7/SoftTeacher/configs/soft_teacher/base.py"
classes = ('polyp',)
num_classes=1
fold = 1
percent = 5
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        sup=dict(
            type="CocoDataset",
            classes=classes,
            ann_file="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/Semi_PolypsSet/semi_1_labels/train2019.${fold}@${percent}.json",
            img_prefix="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/PolypsSet/train2019/images/",
        ),
        unsup=dict(
            type="CocoDataset",
            classes=classes,
            ann_file="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/Semi_PolypsSet/semi_1_labels/train2019.${fold}@${percent}-unlabeled.json",
            img_prefix="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/PolypsSet/train2019/images/",
        ),
    ),
    val=dict(
        type="CocoDataset",
        classes=classes,
        ann_file="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/Semi_PolypsSet/valid_1label.json",
        img_prefix="/home/nguyen.thanh.huyenb/yolov7/PolypsSet/PolypsSet/valid/images",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)


lr_config = dict(step=[4400, 5600])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=10)
evaluation = dict(interval=100)
work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"

log_config = dict(
    interval=20,
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
