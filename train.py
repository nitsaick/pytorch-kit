import datetime
import os
import shutil
import sys
from optparse import OptionParser

import torch
import torch.nn as nn

import dataset
import network
import utils.checkpoint as cp
from dataset.transform import random_flip_transform, medical_transform, real_world_transform
from train_seg import train_seg
from utils.switch import *


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=1, type='int', help='number of epochs')
    parser.add_option('--batch-size', dest='batch_size', default=1, type='int', help='batch size')
    parser.add_option('--lr', dest='lr', default=0.0001, type='float', help='learning rate')
    parser.add_option('--gpu', dest='gpu', default=0, type='int', help='gpu of index, -1 is cpu')
    parser.add_option('--log-root', dest='log_root', default='~/runs/', help='tensorboard log root')
    parser.add_option('--log-name', dest='log_name', default=None, help='tensorboard log name')
    parser.add_option('--resume-file', dest='resume_file', default=False, help='resume checkpoint file')
    parser.add_option('--dataset-root', dest='dataset_root', default='~/dataset/', help='dataset root')
    parser.add_option('--dataset-name', dest='dataset_name', default='SpineSeg',
                      help='dataset name: SpineSeg, xVertSeg, VOC2012Seg')
    parser.add_option('--shuffle', dest='shuffle', default=False, help='dataset shuffle')
    parser.add_option('--network', dest='network_name', default='UNet',
                      help='network name: UNet, R2UNet, AttUNet, AttR2UNet, IDANet, TUNet')
    parser.add_option('--base-ch', dest='base_ch', default=64, type='int', help='base channels')
    parser.add_option('--eval-epoch', dest='eval_epoch', default=1, type='int', help='eval epoch')
    parser.add_option('--pretrained', dest='pretrained', default=False, help='pretrained')
    
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    
    device = 'cpu'
    if args.gpu != -1:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    if args.log_name is None:
        args.log_name = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    
    log_dir = os.path.expanduser(os.path.join(args.log_root, args.log_name))
    dataset_root = os.path.expanduser(os.path.join(args.dataset_root, args.dataset_name))
    
    
    
    for case in switch(args.dataset_name):
        if case('SpineSeg'):
            dataset = dataset.SpineSeg(root=dataset_root, valid_rate=0.2,
                                       train_transform=medical_transform(), valid_transform=None)
            criterion = nn.CrossEntropyLoss()
            break
        
        if case('xVertSeg'):
            dataset = dataset.xVertSeg(root=dataset_root, valid_rate=0.25,
                                       train_transform=random_flip_transform, valid_transform=None)
            criterion = nn.CrossEntropyLoss()
            break
        
        if case('VOC2012Seg'):
            dataset = dataset.VOC2012Seg(root=dataset_root,
                                         train_transform=real_world_transform(output_size=256, type='train'),
                                         valid_transform=real_world_transform(output_size=256, type='valid'))
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            break
        
        if case('Cityscapes'):
            dataset = dataset.Cityscapes(root=dataset_root,
                                         train_transform=real_world_transform(output_size=256, type='train'),
                                         valid_transform=real_world_transform(output_size=256, type='valid'))
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            break
        
        if case():
            raise NotImplementedError('Unknown dataset name.')
    
    checkpoint_dir = os.path.join(log_dir, 'checkpoint')
    
    for case in switch(args.network_name):
        if case('UNet'):
            net = network.UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('R2UNet'):
            net = network.R2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('AttUNet'):
            net = network.AttUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('AttR2UNet'):
            net = network.AttR2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('IDANet'):
            net = network.IDANet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('TUNet'):
            net = network.Tunable_UNet(in_channels=dataset.img_channels, n_classes=dataset.num_classes,
                                       depth=5, wf=6, padding=True, batch_norm=True, up_mode='upconv')
            break
        
        if case('AlbuNet'):
            net = network.AlbuNet(num_classes=dataset.num_classes, num_filters=32,
                                  pretrained=True, is_deconv=True)
            break
        
        if case('DeepLab'):
            net = network.DeepLab(backbone='resnet', output_stride=16, num_classes=dataset.num_classes,
                                  sync_bn=True, freeze_bn=False, pretrained=args.pretrained)
            break
        
        if case():
            raise NotImplementedError('Unknown network name.')
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    start_epoch = 0
    if args.resume_file:
        try:
            if isinstance(args.resume_file, str) and os.path.isfile(args.resume_file):
                net, optimizer, last_epoch = cp.load_params(args.resume_file, net, optimizer, device)
            elif args.resume_file == 'True':
                net, optimizer, last_epoch = cp.load_params(checkpoint_dir, net, optimizer, device, latest=True)
        except FileNotFoundError:
            last_epoch = -1
        start_epoch = last_epoch + 1
    else:
        def check(dir_):
            yes = {'yes', 'y', 'ye', ''}
            no = {'no', 'n'}
            choice = input(msg)
            if choice in yes:
                shutil.rmtree(dir_)
                os.makedirs(dir_)
                return True
            elif choice in no:
                return False
        
        
        if os.path.exists(log_dir):
            files = os.listdir(log_dir)
            if len(files) >= 3:
                msg = '"{}" has old log file. Do you want to delete? (y/n) '.format(log_dir)
                if not check(log_dir): sys.exit(0)
        else:
            os.makedirs(log_dir)
        
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            if len(files):
                msg = '"{}" has old checkpoint file. Do you want to delete? (y/n) '.format(checkpoint_dir)
                if not check(checkpoint_dir): sys.exit(0)
        else:
            os.makedirs(checkpoint_dir)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )
    
    try:
        torch.cuda.empty_cache()
        train_seg(net=net, dataset=dataset, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                  epoch_num=args.epochs, batch_size=args.batch_size, device=device, checkpoint_dir=checkpoint_dir,
                  start_epoch=start_epoch, log_dir=log_dir, eval_epoch=args.eval_epoch)
    
    except KeyboardInterrupt:
        try:
            latest_checkpoint = torch.load(cp.find_latest())
            epoch = latest_checkpoint['epoch']
        except FileNotFoundError:
            epoch = 0
        
        cp_path = os.path.join(checkpoint_dir, 'INTERRUPTED.pth')
        cp.save(epoch, net, optimizer, cp_path)
        
        print('\nSaved interrupt: {}'.format(cp_path))
        sys.exit(0)
