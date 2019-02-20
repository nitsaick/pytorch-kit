import datetime
import os

import torch
import torch.nn as nn
from ruamel import yaml

import dataset
import network
from dataset.transform import random_flip_transform, medical_transform, real_world_transform
from utils.switch import *

__all__ = ['cfg_check', 'gpu_check', 'gpu_check', 'log_name_check',
           'get_transform', 'get_dataset', 'get_criterion', 'get_net',
           'get_optimizer', 'get_scheduler', 'eval_func_check']

def cfg_check(cfg):
    def check(cfg, default_cfg):
        if default_cfg is not None:
            for k, v in default_cfg.items():
                if isinstance(v, dict):
                    check(cfg[k], default_cfg[k])
                elif cfg.get(k) is None:
                    cfg[k] = default_cfg[k]
        else:
            cfg = None

    defalut_file = './configs/default.yml'
    with open(defalut_file) as fp:
        default_cfg = yaml.load(fp, Loader=yaml.Loader)

    for k, v in default_cfg.items():
        if k == 'training':
            check(cfg[k], default_cfg[k])
        elif k == 'transform':
            if cfg[k]['train']['name'] is not None:
                check(cfg[k]['train'], default_cfg[k][cfg[k]['train']['name'].lower()]['train'])
            if cfg[k]['valid']['name'] is not None:
                check(cfg[k]['valid'], default_cfg[k][cfg[k]['valid']['name'].lower()]['valid'])
        else:
            check(cfg[k], default_cfg[k][cfg[k]['name'].lower()])


def gpu_check(cfg):
    if cfg['training']['device']['name'] == 'gpu':
        assert torch.cuda.is_available(), 'GPU is not available'
        assert torch.cuda.device_count() >= len(cfg['training']['device']['ids'])
        for id in cfg['training']['device']['ids']:
            assert id < torch.cuda.device_count()


def log_name_check(cfg):
    if cfg['training'].get('log_name') is None:
        cfg['training']['log_name'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_transform(cfg, type):
    transform_params = {k: v for k, v in cfg['transform'][type].items() if k != 'name'}
    if cfg['transform'][type]['name'] is None:
        transform = None
    else:
        for case in switch(cfg['transform'][type]['name'].lower()):
            if case('medical_transform'):
                transform = medical_transform(**transform_params)
                break

            if case('real_world_transform'):
                transform = real_world_transform(**transform_params)
                break

            if case('random_flip_transform'):
                transform = random_flip_transform
                break

            if case():
                raise NotImplementedError('Unknown transform name.')

    return transform


def get_dataset(cfg, dataset_root, train_transform, valid_transform):
    assert os.path.exists(dataset_root), '{} does not exist'.format(dataset_root)

    for case in switch(cfg['dataset']['name'].lower()):
        if case('spineseg'):
            dataset_ = dataset.SpineSeg(root=dataset_root, valid_rate=0.2,
                                        train_transform=train_transform,
                                        valid_transform=valid_transform)
            break

        if case('xvertseg'):
            dataset_ = dataset.xVertSeg(root=dataset_root, valid_rate=0.25,
                                        train_transform=train_transform,
                                        valid_transform=valid_transform)
            break

        if case('voc2012seg'):
            dataset_ = dataset.VOC2012Seg(root=dataset_root,
                                          train_transform=train_transform,
                                          valid_transform=valid_transform)
            break

        if case('cityscapes'):
            dataset_ = dataset.Cityscapes(root=dataset_root,
                                          train_transform=train_transform,
                                          valid_transform=valid_transform)
            break

        if case():
            raise NotImplementedError('Unknown dataset name.')

    return dataset_


def get_criterion(cfg):
    criterion_params = {k: v for k, v in cfg['loss'].items() if k != 'name'}
    for case in switch(cfg['loss']['name'].lower()):
        if case('cross_entropy'):
            criterion = nn.CrossEntropyLoss(**criterion_params)
            break

        if case():
            raise NotImplementedError('Unknown loss name.')

    return criterion


def get_net(cfg, dataset):
    net_params = {k: v for k, v in cfg['net'].items() if k != 'name'}
    for case in switch(cfg['net']['name'].lower()):
        if case('unet'):
            net = network.UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, **net_params)
            break

        if case('r2unet'):
            net = network.R2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, **net_params)
            break

        if case('attunet'):
            net = network.AttUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, **net_params)
            break

        if case('attr2unet'):
            net = network.AttR2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, **net_params)
            break

        if case('idanet'):
            net = network.IDANet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, **net_params)
            break

        if case('tunet'):
            net = network.Tunable_UNet(in_channels=dataset.img_channels, n_classes=dataset.num_classes, **net_params)
            break

        if case('albunet'):
            net = network.AlbuNet(num_classes=dataset.num_classes, **net_params)
            break

        if case('deeplab'):
            net = network.DeepLab(num_classes=dataset.num_classes, **net_params)
            break

        if case():
            raise NotImplementedError('Unknown network name.')

    return net


def get_optimizer(cfg, net):
    optimizer_params = {k: v for k, v in cfg['optimizer'].items() if k != 'name'}
    for case in switch(cfg['optimizer']['name'].lower()):
        if case('sgd'):
            optimizer = torch.optim.SGD(net.parameters(), **optimizer_params)
            break

        if case('adam'):
            optimizer = torch.optim.Adam(net.parameters(), **optimizer_params)
            break

        if case():
            raise NotImplementedError('Unknown optimizer name.')

    return optimizer


def get_scheduler(cfg, optimizer):
    scheduler_params = {k: v for k, v in cfg['scheduler'].items() if k != 'name'}
    for case in switch(cfg['scheduler']['name'].lower()):
        if case('reduce_lr_on_plateau'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
            break

        if case():
            raise NotImplementedError('Unknown scheduler name.')

    return scheduler


def eval_func_check(cfg, dataset):
    if cfg['training'].get(['eval_func']) is None or cfg['training']['eval_func'] is None:
        if dataset.num_classes <= 2:
            cfg['training']['eval_func'] = 'dc'
        else:
            cfg['training']['eval_func'] = 'mIoU'
