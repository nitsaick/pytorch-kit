import os

import torch
from tensorboardX import SummaryWriter

import checkpoint as cp
import visualize
from metrics import Evaluator


def train_multidim(net, dataset, optimizer, scheduler, criterion, epoch_num=5, batch_size=1, device='cpu',
                   checkpoint_dir='./checkpoint/', start_epoch=0, log_dir='./runs/'):
    net = net.to(device)
    train_loader, valid_loader = dataset.get_dataloader(batch_size)

    train_logger_root = os.path.join(log_dir, 'train')
    valid_logger_root = os.path.join(log_dir, 'valid')
    train_logger = SummaryWriter(train_logger_root)
    valid_logger = SummaryWriter(valid_logger_root)

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

    fig = visualize.create_fig((1, 3))

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
            outputs = net(imgs)

            if batch_index == 0:
                vis_imgs = dataset.batch_visualize_transform(imgs, 'image')
                vis_labels = dataset.batch_visualize_transform(labels, 'label')
                vis_outputs = dataset.batch_visualize_transform(outputs, 'predict')
                visualize.imshow(imgs=(vis_imgs[0], vis_labels[0], vis_outputs[0]), main_title='Train', shape=(1, 3),
                                 sub_title=('image', 'label', 'predict'), transpose=True, fig=fig)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_index + 1) % 10 == 0 or batch_index + 1 == batch_num:
                print('Epoch: {:3d} | Batch: {:5d}/{:<5d} | loss: {:.5f}'
                      .format(epoch + 1, batch_index + 1, batch_num, loss))

        epoch_loss /= batch_num
        scheduler.step(epoch_loss)

        print('Avg loss:  {:.5f}'.format(epoch_loss))
        train_logger.add_scalar('loss', epoch_loss, epoch)

        # Evaluation
        net.eval()
        torch.set_grad_enabled(False)
        evaluator = Evaluator(dataset.num_class)
        for batch_index, (imgs, labels) in enumerate(train_loader):
            # optimizer.zero_grad()
            imgs = imgs.to(device)
            outputs = net(imgs)
            evaluator.add_batch(outputs.cpu().detach().numpy().argmax(axis=1), labels.cpu().detach().numpy())

        train_acc = evaluator.Mean_Intersection_over_Union()
        print('Train data avg acc:  {:.5f}'.format(train_acc))
        train_logger.add_scalar('acc', train_acc, epoch)

        # Validation
        evaluator.reset()
        for batch_index, (imgs, labels) in enumerate(valid_loader):
            # optimizer.zero_grad()
            imgs = imgs.to(device)
            outputs = net(imgs)
            evaluator.add_batch(outputs.cpu().detach().numpy().argmax(axis=1), labels.cpu().detach().numpy())

            # Visualize first image
            if batch_index == 0:
                vis_imgs = dataset.batch_visualize_transform(imgs, 'image')
                vis_labels = dataset.batch_visualize_transform(labels, 'label')
                vis_outputs = dataset.batch_visualize_transform(outputs, 'predict')
                visualize.imshow(imgs=(vis_imgs[0], vis_labels[0], vis_outputs[0]), main_title='Valid', shape=(1, 3),
                                 sub_title=('image', 'label', 'predict'), transpose=True, fig=fig)
                valid_logger.add_image('input', imgs, epoch)
                valid_logger.add_image('label', vis_labels, epoch)
                valid_logger.add_image('output', vis_outputs, epoch)

        valid_acc = evaluator.Mean_Intersection_over_Union()
        print('Valid data avg acc:  {:.5f}'.format(valid_acc))
        valid_logger.add_scalar('acc', valid_acc, epoch)

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
