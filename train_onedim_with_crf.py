import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import utils.checkpoint as cp
import utils.evaluation as eval
from utils.crf import dense_crf_batch
from utils.visualize import imshow


def train_onedim_with_crf(net, dataset, optimizer, scheduler, criterion, epoch_num=5, batch_size=1, device='cpu',
                          checkpoint_dir='./checkpoint/', start_epoch=0, log_dir='./runs/'):
    net = net.to(device)
    train_loader, valid_loader, _ = dataset.get_dataloader(batch_size)

    train_logger_root = os.path.join(log_dir, 'train')
    valid_logger_root = os.path.join(log_dir, 'valid')
    crf_logger_root = os.path.join(log_dir, 'crf')

    train_logger = SummaryWriter(train_logger_root)
    valid_logger = SummaryWriter(valid_logger_root)
    crf_logger = SummaryWriter(crf_logger_root)

    print('{:-^47s}'.format(' Start training '))
    msg = '''
    Net: {}
    Epochs: {}
    Batch size: {}
    Learning rate: {}
    Training size: {}
    Validation size: {}
    Device: {}
        '''.format(net.__class__.__name__, epoch_num, batch_size, optimizer.param_groups[0]['lr'],
                   len(train_loader.sampler),
                   len(valid_loader.sampler), str(device))

    print(msg)
    train_logger.add_text('detail', msg)

    best_acc = 0.0
    best_epoch = 0
    thres = 0.5

    for epoch in range(start_epoch, epoch_num):
        epoch_str = ' Epoch {}/{} '.format(epoch + 1, epoch_num)
        print('{:-^47s}'.format(epoch_str))

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print('Learning rate: {}'.format(lr))
            train_logger.add_scalar('lr', lr, epoch)
            break

        net.train()
        torch.set_grad_enabled(True)
        batch_num = len(train_loader)
        epoch_loss = 0.0

        # Training
        for batch_index, (imgs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)

            output = net(imgs)
            loss = criterion(output, labels)

            if batch_index == 0:
                imgs, labels, output = dataset.batch_visualize_transform(img=imgs, label=labels, pred=output)
                imshow('Train', (imgs[0][0], labels[0][0], output[0][0]), cmap='gray')

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_index + 1) % 10 == 0 or batch_index + 1 == batch_num:
                print('Epoch: {:3d} | Batch: {:5d}/{:<5d} | loss: {:.5f}'.format(epoch + 1, batch_index + 1, batch_num,
                                                                                 loss))

        epoch_loss /= batch_num
        scheduler.step(epoch_loss)

        print('Avg loss:  {:.5f}'.format(epoch_loss))
        train_logger.add_scalar('loss', epoch_loss, epoch)

        # Evaluation
        net.eval()
        torch.set_grad_enabled(False)

        train_acc = 0.0
        train_batch_num = 0
        for batch_index, (imgs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            output = net(imgs)
            output = torch.sigmoid(output)

            train_acc += eval.dice_coef(output, labels, thres)
            train_batch_num += 1

        train_acc /= train_batch_num
        print('Train data avg acc:  {:.5f}'.format(train_acc))
        train_logger.add_scalar('acc', train_acc, epoch)

        valid_acc = 0.0
        valid_batch_num = 0

        crf_acc = 0.0
        for batch_index, (imgs, labels) in enumerate(valid_loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            output = net(imgs)
            output = torch.sigmoid(output)

            valid_acc += eval.dice_coef(output, labels, thres)
            valid_batch_num += 1

            # Visualize first image
            if batch_index == 0:
                imgs, labels, output = dataset.batch_visualize_transform(img=imgs, label=labels, pred=output)
                imshow('Train', (imgs[0][0], labels[0][0], output[0][0]), cmap='gray')
                output = eval.normalize(output)
                thres_img = eval.post_proc(output, thres)
                valid_logger.add_image('image', imgs, epoch)
                valid_logger.add_image('output', output, epoch)
                valid_logger.add_image('threshold', thres_img, epoch)
                valid_logger.add_image('label', labels, epoch)

            img = imgs.cpu().detach().numpy() * 256
            imgs_color = np.concatenate((img, img, img), axis=1).astype(np.uint8)

            outputs = output.cpu().detach().numpy()
            bg = 1 - outputs
            outputs = np.concatenate((bg, outputs), axis=1)

            outputs_crf = dense_crf_batch(imgs_color, outputs)[:, 1]
            outputs_crf = np.expand_dims(outputs_crf, axis=1)

            crf_acc += eval.dice_coef(outputs_crf, labels, thres)

            # Visualize first image
            if batch_index == 0:
                imgs, labels, outputs_crf = dataset.batch_visualize_transform(img=imgs, label=labels, pred=outputs_crf)
                imshow('Train', (imgs[0][0], labels[0][0], outputs_crf[0][0]), cmap='gray')
                output = eval.normalize(outputs_crf)
                thres_img = eval.post_proc(output, thres)
                crf_logger.add_image('output', output, epoch)
                crf_logger.add_image('threshold', thres_img, epoch)
                crf_logger.add_image('label', labels, epoch)

        valid_acc /= valid_batch_num
        print('Valid data avg acc:  {:.5f}'.format(valid_acc))
        valid_logger.add_scalar('acc', valid_acc, epoch)

        crf_acc /= valid_batch_num
        print('Valid data with CRF avg acc:  {:.5f}'.format(crf_acc))
        crf_logger.add_scalar('acc', crf_acc, epoch)

        if (epoch + 1) % 10 == 0:
            checkpoint_filename = 'cp_{:03d}.pth'.format(epoch + 1)
            cp.save(epoch, net, optimizer, os.path.join(checkpoint_dir, checkpoint_filename))

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            checkpoint_filename = 'best.pth'
            cp.save(epoch, net, optimizer, os.path.join(checkpoint_dir, checkpoint_filename))
            print('Update best acc!')
            valid_logger.add_scalar('best epoch', epoch, epoch)

        print('Best epoch: {:3d}    |    Best valid acc: {:.5f}\n'.format(best_epoch + 1, best_acc))

    train_logger.close()
    valid_logger.close()

    return best_acc
