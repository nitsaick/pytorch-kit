import os

import torch

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
    
    net = net.to(device)
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
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, epoch_num):
        epoch_str = ' Epoch {}/{} '.format(epoch + 1, epoch_num)
        print('{:-^47s}'.format(epoch_str))
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        print('Learning rate: {}'.format(lr))
        
        # Training phase
        net.train()
        torch.set_grad_enabled(True)
        loss = training(criterion, dataset, device, epoch, net, optimizer, scheduler, train_loader)
        print('loss:  {:.5f}'.format(loss))
        
        if (epoch + 1) % eval_epoch == 0:
            net.eval()
            torch.set_grad_enabled(False)
            
            # Evaluation phase
            train_acc = evaluation(dataset, device, net, train_loader, eval_func)
            print('Train data avg acc:  {:.5f}'.format(train_acc))
            
            # Validation phase
            valid_acc = validation(dataset, device, net, valid_loader, eval_func)
            print('Valid data avg acc:  {:.5f}'.format(valid_acc))
            
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch
                checkpoint_filename = 'best.pth'
                cp.save(epoch, net, optimizer, os.path.join(checkpoint_dir, checkpoint_filename))
                print('Update best acc!')
            
            print('Best epoch: {:3d}    |    Best valid acc: {:.5f}\n'.format(best_epoch + 1, best_acc))
        
        if (epoch + 1) % checkpoint_save_epoch == 0:
            checkpoint_filename = 'cp_{:03d}.pth'.format(epoch + 1)
            cp.save(epoch, net, optimizer, os.path.join(checkpoint_dir, checkpoint_filename))


def training(criterion, dataset, device, epoch, net, optimizer, scheduler, train_loader):
    batch_num = len(train_loader)
    epoch_loss = 0.0
    
    for batch_index, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = net(imgs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_index == 0:
            outputs = outputs.cpu().detach().numpy().argmax(axis=1)
            imgs, labels, outputs = dataset.vis_transform(imgs, labels, outputs)
            imshow(title='Train', imgs=(imgs[0], labels[0], outputs[0]), shape=(1, 3),
                   subtitle=('image', 'label', 'predict'))
        
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == batch_num:
            print('Epoch: {:3d} | Batch: {:5d}/{:<5d} | loss: {:.5f}'
                  .format(epoch + 1, batch_index + 1, batch_num, loss.item()))
    scheduler.step(loss.item())
    
    return loss.item()


def evaluation(dataset, device, net, train_loader, eval_func):
    evaluator = Evaluator(dataset.num_classes)
    for batch_index, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        outputs = net(imgs)
        
        outputs = outputs.cpu().detach().numpy().argmax(axis=1)
        labels = labels.cpu().detach().numpy()
        evaluator.add_batch(outputs, labels)
    
    train_acc = evaluator.get_acc(eval_func)
    return train_acc


def validation(dataset, device, net, valid_loader, eval_func):
    evaluator = Evaluator(dataset.num_classes)
    for batch_index, (imgs, labels) in enumerate(valid_loader):
        imgs = imgs.to(device)
        outputs = net(imgs)
        
        outputs = outputs.cpu().detach().numpy().argmax(axis=1)
        labels = labels.cpu().detach().numpy()
        evaluator.add_batch(outputs, labels)
        
        if batch_index == 0:
            imgs, labels, outputs = dataset.vis_transform(imgs, labels, outputs)
            imshow(title='Valid', imgs=(imgs[0], labels[0], outputs[0]), shape=(1, 3),
                   subtitle=('image', 'label', 'predict'))
    
    valid_acc = evaluator.get_acc(eval_func)
    return valid_acc
