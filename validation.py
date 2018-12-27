import os
import shutil
import sys
from optparse import OptionParser

import torch
import torch.nn as nn

import checkpoint as cp
from SpineImg import SpineImg
from train import train_model
from unet import UNet


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=2, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-r', '--resume', dest='resume',
                      default=False, help='resume model')
    parser.add_option('-d', '--dataset-root', dest='root',
                      default='./data/', help='dataset root')
    parser.add_option('-k', '--cross-valid-k', dest='k',
                      default=5, type='int', help='cross valid k')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    device = 'cpu'
    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint_root = './cross_valid_checkpoint/'
    if not args.resume:
        files = os.listdir(checkpoint_root)
        if len(files):
            yes = {'yes', 'y', 'ye', ''}
            no = {'no', 'n'}
            msg = '"{}" has old checkpoint file. Do you want to delete? (y/n) '.format(checkpoint_root)
            choice = input(msg)
            if choice in yes:
                shutil.rmtree(checkpoint_root)
                os.mkdir(checkpoint_root)
            elif choice in no:
                sys.exit(0)

    dataset = SpineImg(args.root, resume=args.resume is not False, cross_valid=True, cross_valid_k=args.k)

    all_acc = []

    for step in range(args.k):
        net = UNet(1, 1)
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.lr
                                     )

        checkpoint_root = './cross_valid_checkpoint/' + str(step + 1) + '/'
        if not os.path.isdir(checkpoint_root):
            os.mkdir(checkpoint_root)

        start_epoch = 0

        if args.resume:
            checkpoint = cp.load_file(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print('=> loaded checkpoint "{}" (epoch {})'.format(args.resume, start_epoch + 1))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True,
            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
        )

        criterion = nn.BCEWithLogitsLoss()

        dataset.cross_valid_step(step)

        log_root = 'runs/exp-{}/'.format(step + 1)

        msg = ' Step {}/{} '.format(step + 1, args.k)
        print('{:-^41s}'.format(msg))
        acc = train_model(net=net, dataset=dataset, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                          epoch_num=args.epochs, batch_size=args.batch_size, device=device,
                          checkpoint_root=checkpoint_root,
                          start_epoch=start_epoch, log_root=log_root)
        all_acc.append(acc)

    avg_acc = 0.0
    for i in all_acc:
        avg_acc += i
    avg_acc /= args.k
    print('cross valid acc: {}'.format(avg_acc))
