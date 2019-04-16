import pathlib
import shutil
import sys
from argparse import ArgumentParser

import apex
import torch
from apex.parallel import DistributedDataParallel
from ruamel import yaml
from tensorboardX import SummaryWriter

import utils.checkpoint as cp
from utils.cfg_reader import *
from utils.func import *
from utils.metrics import Evaluator
from utils.vis import imshow


class Trainer:
    def __init__(self, net, dataset, optimizer, scheduler, criterion, start_epoch, log_dir, enable_log, cfg):
        self.net = net
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.log_dir = log_dir
        self.enable_log = enable_log

        self.checkpoint_dir = os.path.join(log_dir, 'checkpoint')
        self.epoch_num = cfg['training']['epoch']
        self.start_epoch = start_epoch
        self.batch_size = cfg['training']['batch_size']
        self.eval_epoch_interval = cfg['training']['eval_epoch_interval']
        self.checkpoint_epoch_interval = cfg['training']['checkpoint_epoch_interval']
        self.visualize_iter_interval = cfg['training']['visualize_iter_interval']
        self.num_workers = cfg['training']['num_workers']
        self.eval_func = cfg['training']['eval_func']
        self.print_summary = cfg['training']['print_summary']

        self.cuda = cfg['training']['device']['name'] == 'gpu'
        if self.cuda:
            self.gpu_ids = cfg['training']['device']['ids']

        if self.enable_log:
            self.logger = SummaryWriter(log_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

    def run(self):
        train_loader, valid_loader, _ = self.dataset.get_dataloader(self.batch_size, self.num_workers, pin_memory=True)

        if self.enable_log:
            self.log_info(valid_loader)

        if self.cuda:
            self.criterion = self.criterion.cuda()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        print('{:-^40s}'.format(' Start training '))
        self.show_train_info(train_loader, valid_loader)

        valid_acc = 0.0
        best_acc = 0.0

        for epoch in range(self.start_epoch, self.epoch_num):
            epoch_str = ' Epoch {}/{} '.format(epoch + 1, self.epoch_num)
            print('{:-^40s}'.format(epoch_str))

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                break
            print('Learning rate: {}'.format(lr))
            if self.enable_log:
                self.logger.add_scalar('lr', lr, epoch)

            # Training phase
            self.net.train()
            torch.set_grad_enabled(True)

            try:
                self.dataset.train_sampler.set_epoch(epoch)
                loss = self.training(train_loader)
            except KeyboardInterrupt:
                if self.enable_log:
                    cp_path = os.path.join(self.checkpoint_dir, 'INTERRUPTED.pth')
                    cp.save(epoch, self.net, self.optimizer, cp_path)
                return

            if enable_log: self.logger.add_scalar('loss', loss, epoch)

            if enable_log and (epoch + 1) % self.eval_epoch_interval == 0:
                self.net.eval()
                torch.set_grad_enabled(False)

                # Evaluation phase
                train_acc = self.evaluation(train_loader, epoch)

                # Validation phase
                valid_acc = self.validation(valid_loader, epoch)

                print('Train data {} acc:  {:.5f}'.format(self.eval_func, train_acc))
                print('Valid data {} acc:  {:.5f}'.format(self.eval_func, valid_acc))

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_epoch = epoch
                    checkpoint_filename = 'best.pth'
                    cp.save(epoch, self.net, self.optimizer, os.path.join(self.checkpoint_dir, checkpoint_filename))
                    print('Update best acc!')
                    self.logger.add_scalar('best epoch', best_epoch, 0)
                    self.logger.add_scalar('best acc', best_acc, 0)

                if (epoch + 1) % self.checkpoint_epoch_interval == 0:
                    checkpoint_filename = 'cp_{:03d}.pth'.format(epoch + 1)
                    cp.save(epoch, self.net.module, self.optimizer, os.path.join(self.checkpoint_dir, checkpoint_filename))

    def show_train_info(self, train_loader, valid_loader):
        if self.cuda:
            device = 'cuda' + str(self.gpu_ids)
        else:
            device = 'cpu'
        msg = 'Net: {}\n'.format(self.net.module.__class__.__name__) + \
              'Dataset: {}\n'.format(self.dataset.__class__.__name__) + \
              'Epochs: {}\n'.format(self.epoch_num) + \
              'Learning rate: {}\n'.format(optimizer.param_groups[0]['lr']) + \
              'Batch size: {}\n'.format(self.batch_size) + \
              'Training size: {}\n'.format(len(train_loader.sampler)) + \
              'Validation size: {}\n'.format(len(valid_loader.sampler)) + \
              'Device: {}\n'.format(device)
        print(msg)

        if enable_log:
            self.logger.add_text('detail', msg)

    def log_info(self, valid_loader):
        # net_summary_text = net_summary(self.net, self.dataset, device='cuda')
        # self.logger.add_text('summary', net_summary_text)
        # with open(os.path.join(self.log_dir, 'summary.txt'), 'w') as file:
        #     file.write(net_summary_text)
        # if self.print_summary:
        #     print(net_summary_text)

        try:
            dummy_input = torch.zeros_like(self.dataset.__getitem__(0)[0])
            dummy_input = dummy_input.view((1,) + dummy_input.shape)
            self.logger.add_graph(net, dummy_input)
        except RuntimeError:
            print('Warning: Cannot export net to ONNX, ignore log graph to tensorboard')

        for batch_idx, (imgs, labels) in enumerate(valid_loader):
            imgs, labels, _ = self.dataset.vis_transform(imgs, labels, None)
            self.logger.add_images('image', imgs, 0)
            self.logger.add_images('label', labels, 0)
            break

    def training(self, train_loader):
        tbar = tqdm(train_loader, ascii=True, desc='train', file=sys.stdout)
        for batch_idx, (imgs, labels) in enumerate(tbar):
            self.optimizer.zero_grad()

            if self.cuda:
                imgs, labels = imgs.cuda(), labels.cuda()

            outputs = self.net(imgs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if self.visualize_iter_interval > 0 and batch_idx % self.visualize_iter_interval == 0:
                outputs = outputs.cpu().detach().numpy().argmax(axis=1)
                imgs, labels, outputs = self.dataset.vis_transform(imgs, labels, outputs)
                imshow(title='Train', imgs=(imgs[0], labels[0], outputs[0]), shape=(1, 3),
                       subtitle=('image', 'label', 'predict'))

            tbar.set_postfix(loss='{:.5f}'.format(loss.item()))
        self.scheduler.step(loss.item())

        return loss.item()

    def evaluation(self, train_loader, epoch):
        evaluator = Evaluator(self.dataset.num_classes)
        tbar = tqdm(train_loader, desc='eval ', ascii=True)
        for batch_idx, (imgs, labels) in enumerate(tbar):
            if self.cuda:
                imgs = imgs.cuda()

            outputs = self.net(imgs)

            outputs = outputs.cpu().detach().numpy().argmax(axis=1)
            labels = labels.cpu().detach().numpy()
            evaluator.add_batch(outputs, labels)

        train_acc = evaluator.eval(self.eval_func)
        evaluator.log_acc(self.logger, epoch, 'train/')
        return train_acc

    def validation(self, valid_loader, epoch):
        evaluator = Evaluator(self.dataset.num_classes)
        tbar = tqdm(valid_loader, desc='valid', ascii=True)
        for batch_idx, (imgs, labels) in enumerate(tbar):
            if self.cuda:
                imgs = imgs.cuda()

            outputs = self.net(imgs)

            np_outputs = outputs.cpu().detach().numpy().argmax(axis=1)
            np_labels = labels.cpu().detach().numpy()
            evaluator.add_batch(np_outputs, np_labels)

            if batch_idx == 0:
                vis_imgs, vis_labels, vis_outputs = self.dataset.vis_transform(imgs, labels, outputs)
                imshow(title='Valid', imgs=(vis_imgs[0], vis_labels[0], vis_outputs[0]), shape=(1, 3),
                       subtitle=('image', 'label', 'predict'))
                self.logger.add_images('output', vis_outputs, epoch)

        valid_acc = evaluator.eval(self.eval_func)
        evaluator.log_acc(self.logger, epoch, 'valid/')
        return valid_acc


def get_args():
    parser = ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument('-c', '--config', type=str, default='./configs/unet_spineseg.yml',
                        help='load config file')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='resume checkpoint')
    parser.add_argument('-l', '--local_rank', default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    if int(os.environ['RANK']) == 0:
        enable_log = True
    else:
        enable_log = False
        sys.stdout = open(os.devnull, 'w')

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.benchmark = True

    if not args.resume:
        cfg_file = args.config
    else:
        cp_file = pathlib.Path(args.resume).expanduser()
        cfg_file = cp_file.parents[1] / 'cfg.yml'
        assert cp_file.is_file() and cfg_file.is_file()

    cfg_file = os.path.expanduser(cfg_file)
    with open(cfg_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.RoundTripLoader)

    cfg_check(cfg)
    print('load config: {}'.format(cfg_file))

    path_file = './configs/path.yml'
    with open(path_file) as fp:
        path_cfg = yaml.load(fp, Loader=yaml.Loader)

    gpu_check(cfg)
    log_name_check(cfg)
    log_dir = os.path.expanduser(os.path.join(path_cfg['log_root'], cfg['training']['log_name']))

    dataset_root = os.path.expanduser(path_cfg[cfg['dataset']['name'].lower()])
    train_transform = get_transform(cfg, 'train')
    valid_transform = get_transform(cfg, 'valid')

    dataset = get_dataset(cfg, dataset_root, train_transform, valid_transform, distributed=True)
    criterion = get_criterion(cfg, dataset)

    net = get_net(cfg, dataset)
    net = apex.parallel.convert_syncbn_model(net)
    net = DistributedDataParallel(net.cuda())

    optimizer = get_optimizer(cfg, net)
    scheduler = get_scheduler(cfg, optimizer)

    eval_func_check(cfg, dataset)


    # init_weights(net, cfg['training']['init_weight_func'])

    def check(path):
        yes = {'yes', 'y', 'ye', ''}
        no = {'no', 'n'}
        choice = input(msg)
        if choice in yes:
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
            return True
        elif choice in no:
            return False


    if not args.resume:
        if enable_log:
            if os.path.exists(log_dir):
                files = os.listdir(log_dir)
                if len(files) >= 3:
                    msg = '"{}" has old log file. Do you want to delete? (y/n) '.format(log_dir)
                    if not check(log_dir):
                        sys.exit(-1)
            else:
                os.makedirs(log_dir)
        start_epoch = 0
    else:
        net, optimizer, start_epoch = cp.load_params(net, optimizer, root=cp_file)

    if enable_log:
        file = os.path.join(log_dir, 'cfg.yml')
        with open(file, 'w', encoding='utf-8') as fp:
            yaml.dump(cfg, fp, default_flow_style=False, Dumper=yaml.RoundTripDumper)

    torch.cuda.empty_cache()
    trainer = Trainer(net, dataset, optimizer, scheduler, criterion, start_epoch, log_dir, enable_log, cfg)
    trainer.run()
