# checkpoint saving
checkpoint_config = dict(interval=20, )
#checkpoint_config = dict(interval=1, CLASSES=['cat', 'dog'])

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')#, log_dir='/home/maya/Pictures/projA_pics/out_files_run_1_resnet/logs')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
