# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.cnn.bricks.conv2d_adaptive_padding import Conv2dAdaptivePadding as conv_layer

from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes)

import torch.nn as nn
import mmcls.models.backbones.efficientnet as efficientnet_orig
import copy
from mmcv.cnn import ConvModule
#TODO change logic
MEAN =  False
REPEAT_FLAG = False
INPUT_SHAPE = 384*2*2

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1) #2

    def forward(self, x):
        # _,batch_size,_,_,_ = x.size()
        #
        # all_batch= []
        # x_i = x[:, :, :, :, :]  # one image in batch
        x_i_flat = x.flatten(start_dim=2)
        x_i_flat = x_i_flat.permute(1, 0, 2)
        #x_i_flat = x_i_flat.flatten(start_dim=1, end_dim=2)
        #print('x_i_flat', x_i_flat.shape)
        #x_i_flat = x_i_flat.permute(1,0,2)
        #print('x_i_flat', x_i_flat.shape)
        queries = self.query(x_i_flat)
        keys = self.key(x_i_flat)
        values = self.value(x_i_flat)
        #print('queries', queries.shape)
        #print('keys', keys.transpose(1, 2).shape)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        #print('scores', scores.shape)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        all_weighted = torch.reshape(weighted, x.shape)

      #  all_weighted = torch.reshape(weighted, (x_i.shape[2], x_i.shape[3], x_i.shape[4]))
        #all_batch.append(weighted)
        # for i in range(batch_size):
        #     x_i = x[:,i,:,:,:] #one image in batch
        #     x_i_flat = x_i.flatten(start_dim=1)
        #     print('x_i_flat', x_i_flat.shape)
        #     queries = self.query(x_i_flat)
        #     keys = self.key(x_i_flat)
        #     values = self.value(x_i_flat)
        #     scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        #     attention = self.softmax(scores)
        #     weighted = torch.bmm(attention, values)
        #     weighted = torch.reshape(weighted, (x_i.shape[2], x_i.shape[3], x_i.shape[4]))
        #     all_batch.append(weighted)

       # all_weighted = torch.stack(all_batch, dim=0)

        return all_weighted

'''
def forward(self, x):
    outs = []
    for i, layer in enumerate(self.layers):
        x = layer(x)
        if i in self.out_indices:
            outs.append(x)

    return tuple(outs)
'''
# THIS IS THE FORWARD FOR THE SEPERATED MODEL:
def forward(self, x): #cont forward and combination
    # Part A
    part_a_output = []
    #part_a_output_1 = self.layers_partA[0](x)
    num_channels = x.shape[1]
    num_layers_per_channel = int(len(self.layers_partA)/num_channels)
    if REPEAT_FLAG:
        num_layers =len(self.layers_partA)
        for index in range(num_channels):
            input = x[:, index, :, :]
            input = input.unsqueeze(dim=1)
            for i in range(num_layers):
                input = self.layers_partA[i](input) #forward from with same filters = share weights

            part_a_output.append(input)
    else:
        for index in range(num_channels):
            input = x[:,index,:,:]
            input = input.unsqueeze(dim=1)
            for i in range(num_layers_per_channel):
                input = self.layers_partA[i+num_layers_per_channel*index](input)

            part_a_output.append(input)


    # Combine Part A outputs with pointwise convolution
    combined_output = torch.stack(part_a_output, dim=0)
    #add attention before combining
    combined_output = self.attention(combined_output)
    combined_output = torch.sum(combined_output, dim=0)
    # if MEAN:
    #     combined_output = torch.mean(combined_output, dim=0) #TODO change to different logic
    # else:
    #     combined_output, _ = torch.max(combined_output, dim=0)  # TODO change to different logic

    # point conv
    # #reshpe tensors
    # part_a_output = [torch.reshape(tensor_i,(32,1,384,1)) for tensor_i in part_a_output]
    # combined_output = torch.cat(part_a_output, dim=1)
    # #print(combined_output.shape)
    # combined_output = self.pw_layer(combined_output)
    # #print(combined_output.shape)
    # combined_output = torch.reshape(combined_output,(32,384,1,1))

    # #reshpe tensors
    # part_a_output = [torch.reshape(tensor_i,(32,1,384,1)) for tensor_i in part_a_output]
   # combined_output = torch.cat(part_a_output, dim=1)
    # #print(combined_output.shape)
    #combined_output = self.attention(combined_output)
    # #print(combined_output.shape)
    #combined_output = torch.reshape(combined_output,(32,384,1,1))


    # Part B
    outs = []
    for i, layer in enumerate(self.layers_partB):
        combined_output = layer(combined_output)
        if REPEAT_FLAG and i+num_layers in self.out_indices:
            outs.append(combined_output)
        if not REPEAT_FLAG and i+num_layers_per_channel in self.out_indices:
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
    tmp_weights = torch.FloatTensor(40,image_num,3,3).uniform_(-1/np.sqrt(3*3*image_num*40),1/np.sqrt(3*3*image_num*40))
    #torch.randn((40,image_num,3,3),requires_grad=True)
    new_layer = conv_layer(image_num,40,kernel_size=(3,3), stride= (2,2),bias=False)
    new_layer.weight = torch.nn.parameter.Parameter(tmp_weights)
    self.layers[0].conv = new_layer
    partA = self.layers[:seperate_block]

    self.layers_partB = self.layers[seperate_block:]
    #change forward of model
    # self.forward = forward
    #duplicate the first part
    self.layers_partA = nn.ModuleList()
    if REPEAT_FLAG:
        self.layers_partA = partA
    else:
        for i in range(num_channels):
            self.layers_partA = self.layers_partA + partA #nn.ModuleList(list(partA)).to('cuda:0')

    #add pointwise conv, reshape the tensors before
    # tmp_weights = torch.FloatTensor(1, 4, 1, 1).uniform_(-1 / np.sqrt(4),
    #                                                               1 / np.sqrt(4))
    # self.pw_layer = conv_layer(4, 1, kernel_size=1, stride=1, bias=False)
    # self.pw_layer.weight = torch.nn.parameter.Parameter(tmp_weights)

    #add attention, reshape the tensors before
    self.attention = SelfAttention(INPUT_SHAPE)
    #self.pw_layer = conv_layer(4, 1, kernel_size=1, stride=1, bias=False)
    #self.pw_layer.weight = torch.nn.parameter.Parameter(tmp_weights)

    # Combine the ModuleLists
    # combined_list = nn.ModuleList()
    # combined_list.extend(layers_partA)
    # combined_list.extend(layers_partB)
    # self.layers = combined_list
    # print('hi')



def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--device', help='device used for training. (Deprecated)')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--ipu-replicas',
        type=int,
        default=None,
        help='num of ipu replicas to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(our_adjustments, image_num, metadata_flag, freeze_flag, train_layers, seperate_block=3, adjust_model_flag=False):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.ipu_replicas is not None:
        cfg.ipu_replicas = args.ipu_replicas
        args.device = 'ipu'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = args.device or auto_select_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)

    model.init_weights()

    if our_adjustments:
      #  adjust_model(model,4)

        if metadata_flag:
            image_num =image_num + 1
        #TODO comment later
        #tmp_weights = torch.FloatTensor(40,image_num,3,3).uniform_(-1/np.sqrt(3*3*image_num*40),1/np.sqrt(3*3*image_num*40))
        #torch.randn((40,image_num,3,3),requires_grad=True)
        #new_layer = conv_layer(image_num,40,kernel_size=(3,3), stride= (2,2),bias=False)
        #new_layer.weight = torch.nn.parameter.Parameter(tmp_weights)
        #model.backbone.layers[0].conv = new_layer
        #TODO return
        if adjust_model_flag:
            efficientnet_orig.EfficientNet.adjust_model = adjust_model
            efficientnet_orig.EfficientNet.forward = forward
            model.backbone.adjust_model(num_channels=4, seperate_block = seperate_block)
        else:
            tmp_weights = torch.FloatTensor(40,image_num,3,3).uniform_(-1/np.sqrt(3*3*image_num*40),1/np.sqrt(3*3*image_num*40))
            torch.randn((40,image_num,3,3),requires_grad=True)
            new_layer = conv_layer(image_num,40,kernel_size=(3,3), stride= (2,2),bias=False)
            new_layer.weight = torch.nn.parameter.Parameter(tmp_weights)
            model.backbone.layers[0].conv = new_layer

    if freeze_flag:
        counter = 1
        for child in model.children():
            for param in child.parameters():
                if train_layers < counter:
                    param.requires_grad = False
                counter = counter+1

    #init w&b
    #wandb.init(project="regular-training")

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # save mmcls version, config file content and class names in
    # runner as meta data
    meta.update(
        dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES))

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device=cfg.device,
        meta=meta)



if __name__ == '__main__':
    our_adjustments =True #replace the first layer with random values
    image_num = 4
    metadata_flag = False
    seperate_block = 6
    adjust_model_flag = True #adjust for separating the model
    main(our_adjustments,image_num,metadata_flag,False,2,seperate_block, adjust_model_flag=adjust_model_flag)

