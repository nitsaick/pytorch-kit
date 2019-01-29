import os

import torch
from tensorboardX import SummaryWriter

import utils.checkpoint as cp
from utils.metrics import Evaluator
from utils.vis import imshow


def train_seg(net, dataset, optimizer, scheduler, criterion, epoch_num=5, batch_size=1, device='cpu',
              checkpoint_dir='./checkpoint/', start_epoch=0, log_dir='./runs/',
              eval_func=None, eval_epoch=1, checkpoint_save_epoch=10):
    if eval_func is None:
        if dataset.num_classes <= 2:
            eval_func = 'dc'
        else:
            eval_func = 'mIoU'
    if eval_epoch < 1:
        eval_epoch = 1
    if checkpoint_save_epoch < 1:
        checkpoint_save_epoch = 10
    
    train_loader, valid_loader, _ = dataset.get_dataloader(batch_size)
    
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

    train_logger_root = os.path.join(log_dir, 'train')
    valid_logger_root = os.path.join(log_dir, 'valid')
    train_logger = SummaryWriter(train_logger_root)
    valid_logger = SummaryWriter(valid_logger_root)
    
    dummy_input = torch.zeros_like(dataset.__getitem__(0)[0])
    dummy_input = dummy_input.view((1,) + dummy_input.shape)
    train_logger.add_graph(net, dummy_input)
    train_logger.add_text('detail', msg)
    
    for batch_idx, (imgs, labels) in enumerate(valid_loader):
        imgs, labels, _ = dataset.vis_transform(imgs, labels, None)
        valid_logger.add_images('image', imgs, 0)
        valid_logger.add_images('label', labels, 0)
        break
    
    net = net.to(device)
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, epoch_num):
        epoch_str = ' Epoch {}/{} '.format(epoch + 1, epoch_num)
        print('{:-^47s}'.format(epoch_str))
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        print('Learning rate: {}'.format(lr))
        train_logger.add_scalar('lr', lr, epoch)
        
        if lr < 1e-7:
            print('Learning rate is less than 1e-7. Stop training.')
            break
        
        # Training phase
        net.train()
        torch.set_grad_enabled(True)
        loss = training(criterion, dataset, device, epoch, net, optimizer, scheduler, train_loader)
        print('loss:  {:.5f}'.format(loss))
        train_logger.add_scalar('loss', loss, epoch)
        
        if (epoch + 1) % eval_epoch == 0:
            net.eval()
            torch.set_grad_enabled(False)
            
            # Evaluation phase
            train_acc = evaluation(dataset, device, net, train_loader, eval_func)
            print('Train data avg acc:  {:.5f}'.format(train_acc))
            train_logger.add_scalar('acc', train_acc, epoch)
            
            # Validation phase
            valid_acc = validation(dataset, device, net, valid_loader, eval_func, epoch, valid_logger)
            print('Valid data avg acc:  {:.5f}'.format(valid_acc))
            valid_logger.add_scalar('acc', valid_acc, epoch)
            
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch
                checkpoint_filename = 'best.pth'
                cp.save(epoch, net, optimizer, os.path.join(checkpoint_dir, checkpoint_filename))
                print('Update best acc!')
                valid_logger.add_scalar('best epoch', best_epoch, 0)
            
            print('Best epoch: {:3d}    |    Best valid acc: {:.5f}\n'.format(best_epoch + 1, best_acc))
        
        if (epoch + 1) % checkpoint_save_epoch == 0:
            checkpoint_filename = 'cp_{:03d}.pth'.format(epoch + 1)
            cp.save(epoch, net, optimizer, os.path.join(checkpoint_dir, checkpoint_filename))


def training(criterion, dataset, device, epoch, net, optimizer, scheduler, train_loader):
    batch_num = len(train_loader)
    epoch_loss = 0.0
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = net(imgs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx == 0:
            outputs = outputs.cpu().detach().numpy().argmax(axis=1)
            imgs, labels, outputs = dataset.vis_transform(imgs, labels, outputs)
            imshow(title='Train', imgs=(imgs[0], labels[0], outputs[0]), shape=(1, 3),
                   subtitle=('image', 'label', 'predict'))
        
        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == batch_num:
            print('Epoch: {:3d} | Batch: {:5d}/{:<5d} | loss: {:.5f}'
                  .format(epoch + 1, batch_idx + 1, batch_num, loss.item()))
    scheduler.step(loss.item())
    
    return loss.item()


def evaluation(dataset, device, net, train_loader, eval_func):
    evaluator = Evaluator(dataset.num_classes)
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        outputs = net(imgs)
        
        outputs = outputs.cpu().detach().numpy().argmax(axis=1)
        labels = labels.cpu().detach().numpy()
        evaluator.add_batch(outputs, labels)
    
    train_acc = evaluator.eval(eval_func)
    return train_acc


def validation(dataset, device, net, valid_loader, eval_func, epoch, valid_logger):
    evaluator = Evaluator(dataset.num_classes)
    for batch_idx, (imgs, labels) in enumerate(valid_loader):
        imgs = imgs.to(device)
        outputs = net(imgs)
        
        np_outputs = outputs.cpu().detach().numpy().argmax(axis=1)
        np_labels = labels.cpu().detach().numpy()
        evaluator.add_batch(np_outputs, np_labels)
        
        if batch_idx == 0:
            vis_imgs, vis_labels, vis_outputs = dataset.vis_transform(imgs, labels, outputs)
            imshow(title='Valid', imgs=(vis_imgs[0], vis_labels[0], vis_outputs[0]), shape=(1, 3),
                   subtitle=('image', 'label', 'predict'))
            valid_logger.add_images('output', vis_outputs, epoch)
    
    valid_acc = evaluator.eval(eval_func)
    return valid_acc
