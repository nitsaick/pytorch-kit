import datetime
import os
import shutil
import sys
from optparse import OptionParser

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import checkpoint as cp
import network
import test
import visualize
from VOC import VOC2012
from xVertSeg import xVertSeg
from SpineImg import SpineImg
from SpineImg import random_flip_transform
from Tunable_UNet import Tunable_UNet
from metrics import Evaluator

from train_onedim import train_onedim
from train_multidim import train_multidim


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=20,
                      type='int', help='batch size')
    parser.add_option('-l', '--lr', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', dest='gpu', default=0, type='int',
                      help='gpu of index, -1 is cpu')
    parser.add_option('-t', '--log-root', dest='log_root',
                      default='./runs/', help='tensorboard log root')
    parser.add_option('-n', '--log-name', dest='log_name',
                      default=None, help='tensorboard log name')
    parser.add_option('-r', '--resume-file', dest='resume_file',
                      default=False, help='resume checkpoint file')
    parser.add_option('-d', '--dataset-root', dest='dataset_root',
                      default='./data/', help='dataset root')
    parser.add_option('-f', '--dataset-name', dest='dataset_name',
                      default='SpineImg', help='dataset name')
    parser.add_option('-m', '--network', dest='network_name',
                      default='UNet', help='network name')
    parser.add_option('-c', '--base_ch', dest='base_ch', default=64,
                      type='int', help='base channels')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    device = 'cpu'
    if args.gpu != -1:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    if args.log_name is None:
        args.log_name = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    log_dir = os.path.join(args.log_root, args.log_name)
    checkpoint_dir = os.path.join(log_dir, 'checkpoint')

    dataset_root = os.path.join(args.dataset_root, args.dataset_name)

    if args.dataset_name == 'SpineImg':
        dataset = SpineImg(root=dataset_root, transform=random_flip_transform, resume=args.resume_file is not False, shuffle=False, valid_rate=0.2, log_dir=log_dir)
    elif args.dataset_name == 'VOC':
        dataset = VOC2012(root=dataset_root, shuffle=False, valid_rate=0.2, log_dir=log_dir)
    elif args.dataset_name == 'xVertSeg':
        dataset = xVertSeg(root=dataset_root, resume=args.resume_file is not False, shuffle=False, valid_rate=0.25, log_dir=log_dir)

    criterion = {
        'SpineImg': nn.BCEWithLogitsLoss(),
        'VOC': nn.CrossEntropyLoss(ignore_index=255),
        'xVertSeg': nn.BCEWithLogitsLoss()
    }[args.dataset_name]

    net = {
        'UNet': network.U_Net(img_ch=dataset.img_channels, base_ch=args.base_ch, output_ch=dataset.num_class),
        'R2UNet': network.R2U_Net(img_ch=dataset.img_channels, base_ch=args.base_ch, output_ch=dataset.num_class),
        'AttUNet': network.AttU_Net(img_ch=dataset.img_channels, output_ch=dataset.num_class),
        'AttR2UNet': network.R2AttU_Net(img_ch=dataset.img_channels, output_ch=dataset.num_class),
        'IDANet': network.IDANet(img_ch=dataset.img_channels, base_ch=64, output_ch=dataset.num_class),
        'TUNet': Tunable_UNet(in_channels=1, n_classes=1, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upconv'),
        'test': test.unet_CT_single_att_dsv_2D()
    }[args.network_name]

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume_file:
        checkpoint = cp.load_file(args.resume_file)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print('=> loaded checkpoint "{}" (epoch {})'.format(args.resume_file, start_epoch + 1))
    else:
        if (os.path.exists(log_dir)):
            files = os.listdir(log_dir)
            if len(files) > 3:
                yes = {'yes', 'y', 'ye', ''}
                no = {'no', 'n'}
                msg = '"{}" has old log file. Do you want to delete? (y/n) '.format(log_dir)
                choice = input(msg)
                if choice in yes:
                    shutil.rmtree(log_dir)
                    os.makedirs(log_dir)
                elif choice in no:
                    sys.exit(0)
        else:
            os.makedirs(log_dir)

        if (os.path.exists(checkpoint_dir)):
            files = os.listdir(checkpoint_dir)
            if len(files):
                yes = {'yes', 'y', 'ye', ''}
                no = {'no', 'n'}
                msg = '"{}" has old checkpoint file. Do you want to delete? (y/n) '.format(checkpoint_dir)
                choice = input(msg)
                if choice in yes:
                    shutil.rmtree(checkpoint_dir)
                    os.makedirs(checkpoint_dir)
                elif choice in no:
                    sys.exit(0)
        else:
            os.makedirs(checkpoint_dir)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )

    try:
        if args.dataset_name == 'SpineImg' or args.dataset_name == 'xVertSeg':
            train_onedim(net=net, dataset=dataset, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                         epoch_num=args.epochs, batch_size=args.batch_size, device=device,
                         checkpoint_dir=checkpoint_dir,
                         start_epoch=start_epoch, log_dir=log_dir)
        elif args.dataset_name == 'VOC':
            train_multidim(net=net, dataset=dataset, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                           epoch_num=args.epochs, batch_size=args.batch_size, device=device,
                           checkpoint_dir=checkpoint_dir,
                           start_epoch=start_epoch, log_dir=log_dir)



    except KeyboardInterrupt:
        try:
            latest_checkpoint = cp.load_latest(checkpoint_dir)
            epoch = latest_checkpoint['epoch']
        except FileNotFoundError:
            epoch = 0

        cp_path = os.path.join(checkpoint_dir, 'INTERRUPTED.pth')
        cp.save(epoch, net, optimizer, cp_path)
        print('\nSaved interrupt: {}'.format(cp_path))
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
