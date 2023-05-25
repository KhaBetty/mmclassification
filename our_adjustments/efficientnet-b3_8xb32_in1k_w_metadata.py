import cv2
_base_ = [
    '../configs/_base_/models/efficientnet_b3.py',
    '../configs/_base_/datasets/imagenet_bs32.py',
    '../configs/_base_/schedules/imagenet_bs256.py',
    '../configs/_base_/default_runtime.py',
]
#TODO resized gray images
work_dir = '/home/maya/projA/runs/subpixel_32_4_depth_2_dir_2_metadata' #'/home/maya/projA/runs/original_run_effecientnet' #TODO
dataset_prefix = '/home/maya/Pictures/projA_pics/subpixel_32_4_depth_2_dir_2_metadata' #'/phome/maya/Pictures/projA_pics/dataset_balanced'#TODO
multi_image_flag = True
multi_num= 4 #number of channels in the input
max_epoch_num = 150 #150 #150
shuffle_flag = False
metadata_name = 'direction_matrix.npy'
metadata_flag = True
#load_from = '/home/maya/projA/runs/resized_32_rgb/epoch_50.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b3')
    # ,
    #               init_cfg = dict(type='Pretrained',
    #                 checkpoint='../our_adjustments/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth')
    ,
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

dataset_type = 'AdjustedCustomDataset'

classes = ['cats', 'dogs']  # The category names of your dataset


# #/home/maya/Pictures/projA_pics/dataset'


# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=224),
#     #dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=224),
#     #dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]

#RGB

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', size=224),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=(256, -1)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]



#Grey
img_norm_cfg = dict(
    mean=[114.495]* multi_num + [114.495]*metadata_flag,
    std=[57.6]*multi_num + [57.6]*metadata_flag, to_rgb=False)

train_pipeline = [
    dict(type='LoadMetadataMultiChannelImages', color_type=cv2.IMREAD_GRAYSCALE, metadata=metadata_flag) if multi_image_flag else dict(type='LoadImageFromFile',
                                                                  color_type=cv2.IMREAD_GRAYSCALE),

   # dict(type= 'LoadMultiChannelImages', color_type=cv2.IMREAD_GRAYSCALE,shuffle_flag=shuffle_flag) if multi_image_flag else dict(type='LoadImageFromFile',color_type=cv2.IMREAD_GRAYSCALE),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadMetadataMultiChannelImages', color_type=cv2.IMREAD_GRAYSCALE, metadata=metadata_flag) if multi_image_flag else dict(
        type='LoadImageFromFile', color_type=cv2.IMREAD_GRAYSCALE),

    #dict(type= 'LoadMultiChannelImages', color_type=cv2.IMREAD_GRAYSCALE) if multi_image_flag else dict(type='LoadImageFromFile',color_type=cv2.IMREAD_GRAYSCALE),
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
        pipeline = train_pipeline,
        matadata_file_name = metadata_name
    ),
    val=dict(
        type=dataset_type,
        data_prefix= dataset_prefix +'/validation_set',
        ann_file = None,
        classes=classes,
        pipeline = test_pipeline,
        matadata_file_name = metadata_name
    ),
    test=dict(
        type=dataset_type,
        data_prefix= dataset_prefix+ '/test_set',
        ann_file = None,
        classes=classes,
        pipeline = test_pipeline,
        matadata_file_name = metadata_name
    )
)
evaluation = dict(interval=1, metric='accuracy', metric_options= {'topk': (1, )})
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }


log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook', log_dir=work_dir +'/logs', by_epoch=True),
        # dict(type='MMClsWandbHook',
        #      init_kwargs={
        #          #'entity': "proj_a@walla.com",
        #          'project': "gray_train",
        #          'dir': work_dir #"/home/maya/projA/runs/original_dataset"
        #      },
        #      log_checkpoint=True,
        #      log_checkpoint_metadata=True,
        #      num_eval_images=100)#a multiple of 2

    ])

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) #TODO temporary instead of 0.01

optimizer_config = dict(grad_clip=None)
# learning policy
#lr_config = dict(policy='step', step=[int(max_epoch_num/3),int(2*max_epoch_num/3)] if multi_image_flag else [int(multi_num*max_epoch_num/3),int(multi_num*2*max_epoch_num/3)])#step=[30, 60])#, 90])
lr_config = dict(policy='step', step=[10000])#[int(max_epoch_num/3),int(2*max_epoch_num/3)] )#TODO temporary should be fixed for now

runner = dict(type='EpochBasedRunner', max_epochs= int(max_epoch_num/(multi_num+metadata_flag)))#divide by number of channels because of the way we go through all pics)
#TODO change max
workflow = [('train', 1),('val',1)]

checkpoint_config=dict(interval=int(30/multi_num))