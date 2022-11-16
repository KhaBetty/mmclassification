_base_ = [
    '../configs/_base_/models/resnet50.py', '../configs/_base_/datasets/imagenet_bs32.py',
    '../configs/_base_/schedules/imagenet_bs256.py', '../configs/_base_/default_runtime.py'
]
_deprecation_ = dict(
    expected='resnet50_8xb32_in1k.py',
    reference='https://github.com/open-mmlab/mmclassification/pull/508',
)

work_dir = '/home/maya/projA/runs/original_run_resnet50_cropped_150_epochs_imagenet' #'/home/maya/projA/runs/original_run_effecientnet' #TODO
dataset_prefix ='/home/maya/Pictures/projA_pics/dataset_balanced'
#'/home/maya/Pictures/projA_pics/fixed_pics/resized_64' #'/home/maya/Pictures/projA_pics/dataset_balanced'#TODO
# #/home/maya/Pictures/projA_pics/dataset'


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
classes = ['cats', 'dogs']
# dataset settings
dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]


data = dict(
    train=dict(
        type=dataset_type,
        data_prefix= dataset_prefix + '/training_set',
        ann_file = None,
        classes=classes,
        pipeline = train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix= dataset_prefix +'/validation_set',
        ann_file = None,
        classes=classes,
        pipeline = test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix= dataset_prefix+ '/test_set',
        ann_file = None,
        classes=classes,
        pipeline = test_pipeline
    )
)
evaluation = dict(interval=1, metric='accuracy', metric_options= {'topk': (1, )})

# optimizer
optimizer = dict(type='SGD', lr=0.01)#, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90, 150])
runner = dict(type='EpochBasedRunner', max_epochs= 200)#100)

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir=work_dir +'/logs'),
        dict(type='MMClsWandbHook',
             init_kwargs={
                 #'entity': "proj_a@walla.com",
                 'project': "basic_train",
                 'dir': work_dir #"/home/maya/projA/runs/original_dataset"
             },
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100)#a multiple of 2

    ])