_base_ = '../configs/resnet/resnet18_8xb32_in1k.py'

_deprecation_ = dict(
    expected='../configs/resnet/resnet18_8xb32_in1k.py',
    reference='https://github.com/open-mmlab/mmclassification/pull/508',
)

dataset_type = 'CustomDataset'
classes = ['cat', 'dog']  # The category names of your dataset

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