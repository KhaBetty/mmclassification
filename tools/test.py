# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcv.cnn.bricks.conv2d_adaptive_padding import Conv2dAdaptivePadding as conv_layer
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)
import torch.nn as nn
import mmcls.models.backbones.efficientnet as efficientnet_orig
MEAN = False

def forward(self, x): #cont forward and combination
    # Part A
    part_a_output = []
    #part_a_output_1 = self.layers_partA[0](x)
    num_channels = x.shape[1]
    num_layers_per_channel = int(len(self.layers_partA)/num_channels)
    for index in range(num_channels):
        input = x[:,index,:,:]
        input = input.unsqueeze(dim=1)
        for i in range(num_layers_per_channel):
            input = self.layers_partA[i+num_layers_per_channel*index](input)

        part_a_output.append(input)


    # Combine Part A outputs with pointwise convolution
    combined_output = torch.stack(part_a_output, dim=0)
    #print(combined_output.shape)
    if MEAN:
        combined_output = torch.mean(combined_output, dim=0) #TODO change to different logic
    else:
        combined_output, _ = torch.max(combined_output, dim=0)  # TODO change to different logic


    # Part B
    outs = []
    for i, layer in enumerate(self.layers_partB):
        combined_output = layer(combined_output)
        if i+num_layers_per_channel in self.out_indices:
            outs.append(combined_output)
    #part_b_output = self.layers_partB(combined_output)

    # Return final output
    #print('outs', outs) #TODO
    return tuple(outs) #part_b_output




    # outs = []
    # for i, layer in enumerate(self.layers):
    #     x = layer(x)
    #     if i in self.out_indices:
    #         outs.append(x)
    #
    # return tuple(outs)

def adjust_model(self, seperate_block, num_channels=1): #change the model structure and forward
    image_num=1
    #tmp_weights = torch.FloatTensor(40,image_num,3,3).uniform_(-1/np.sqrt(3*3*image_num*40),1/np.sqrt(3*3*image_num*40))
    #torch.randn((40,image_num,3,3),requires_grad=True)
    new_layer = conv_layer(image_num,40,kernel_size=(3,3), stride= (2,2),bias=False)
   # new_layer.weight = torch.nn.parameter.Parameter(tmp_weights)
    self.layers[0].conv = new_layer
    partA = self.layers[:seperate_block]

    self.layers_partB = self.layers[seperate_block:]
    #change forward of model
    # self.forward = forward
    #duplicate the first part
    self.layers_partA = nn.ModuleList()
    for i in range(num_channels):
        self.layers_partA = self.layers_partA + partA #nn.ModuleList(list(partA)).to('cuda:0')




def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    return args


def main(our_adjustments, image_num, divided_model=False, seperate_block=6):
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    if our_adjustments:
        if divided_model:
            efficientnet_orig.EfficientNet.adjust_model = adjust_model
            efficientnet_orig.EfficientNet.forward = forward
            model.backbone.adjust_model(num_channels=4, seperate_block = seperate_block)
        else:
            model.backbone.layers[0].conv = conv_layer(image_num,40,kernel_size=(3,3), stride= (2,2),bias=False)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    if not distributed:
        model = wrap_non_distributed_model(
            model, device=cfg.device, device_ids=cfg.gpu_ids)
        if cfg.device == 'ipu':
            from mmcv.device.ipu import cfg2options, ipu_model_wrapper
            opts = cfg2options(cfg.runner.get('options_cfg', {}))
            if fp16_cfg is not None:
                model.half()
            model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
            data_loader.init(opts['inference'])
        model.CLASSES = CLASSES
        show_kwargs = args.show_options or {}
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = wrap_distributed_model(
            model,
            device=cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        logger = get_root_logger()
        if args.metrics:
            eval_results = dataset.evaluate(
                results=outputs,
                metric=args.metrics,
                metric_options=args.metric_options,
                logger=logger)
            results.update(eval_results)
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray):
                    v = [round(out, 2) for out in v.tolist()]
                elif isinstance(v, Number):
                    v = round(v, 2)
                else:
                    raise ValueError(f'Unsupport metric type: {type(v)}')
                print(f'\n{k} : {v}')
        if args.out:
            if 'none' not in args.out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    'class_scores': scores,
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class
                }
                if 'all' in args.out_items:
                    results.update(res_items)
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
            print(f'\ndumping results to {args.out}')
            mmcv.dump(results, args.out)


if __name__ == '__main__':
    our_adjustments = True
    image_num = 4
    main(our_adjustments, image_num, divided_model=True, seperate_block=6)
