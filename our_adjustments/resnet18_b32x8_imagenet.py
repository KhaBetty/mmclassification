_base_ = '../configs/resnet/resnet18_8xb32_in1k.py'

_deprecation_ = dict(
    expected='../configs/resnet/resnet18_8xb32_in1k.py',
    reference='https://github.com/open-mmlab/mmclassification/pull/508',
)

dataset_type = 'CustomDataset'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type= 'LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


classes = ['cats', 'dogs']  # The category names of your dataset

dataset_prefix = '/home/maya/Pictures/projA_pics/dataset'

data = dict(
    train=dict(
        type=dataset_type,
        data_prefix= dataset_prefix + '/training_set',
        ann_file = None,
        classes=classes,
        # pipeline = [dict(type='LoadImageFromFile'),
        #         dict(type='Resize', size=256),
        #         dict(type='CenterCrop', crop_size=224),
        #         dict(type='ImageToTensor', keys=['img']),
        #         dict(type='Collect', keys=['img'])]
    ),
    val=dict(
        type=dataset_type,
        data_prefix= dataset_prefix +'/validation_set',
        ann_file = None,
        classes=classes,
        # pipline = [dict(type='LoadImageFromFile'),
        #         dict(type='Resize', size=256),
        #         dict(type='CenterCrop', crop_size=224),
        #         dict(type='ImageToTensor', keys=['img']),
        #         dict(type='Collect', keys=['img'])]
    ),
    test=dict(
        type=dataset_type,
        data_prefix= dataset_prefix+ '/test_set',
        ann_file = None,
        classes=classes,
        # pipline = [dict(type='LoadImageFromFile'),
        #         dict(type='Resize', size=256),
        #         dict(type='CenterCrop', crop_size=224),
        #         dict(type='ImageToTensor', keys=['img']),
        #         dict(type='Collect', keys=['img'])]
    )
)
evaluation = dict(interval=1, metric='accuracy', metric_options= {'topk': (1, )})

# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }

work_dir ='/home/maya/Pictures/projA_pics/out_files_run_1_resnet'

runner = dict(type='EpochBasedRunner', max_epochs=100)