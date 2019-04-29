import pathlib
import shutil
import sys
from argparse import ArgumentParser

import apex
import torch
import torch.distributed as dist
from apex.parallel import DistributedDataParallel
from ruamel import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

import utils.checkpoint as cp
from utils.cfg_reader import *
from utils.func import *
from utils.metrics import Evaluator
from utils.vis import imshow


class Trainer:
    def __init__(self, cfg, resume, distributed, enable_log):
        self.distributed = distributed
        self.enable_log = enable_log
        self.prepare(cfg, resume)

        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoint')
        self.epoch_num = cfg['training']['epoch']
        self.batch_size = cfg['training']['batch_size']
        self.eval_epoch_interval = cfg['training']['eval_epoch_interval']
        self.checkpoint_epoch_interval = cfg['training']['checkpoint_epoch_interval']
        self.visualize_iter_interval = cfg['training']['visualize_iter_interval']
        self.num_workers = cfg['training']['num_workers']
        self.eval_method = cfg['training']['eval_method']
        self.eval_func = cfg['training']['eval_func']
        self.print_summary = cfg['training']['print_summary']

        self.cuda = cfg['training']['device']['name'] == 'gpu'
        if self.cuda:
            self.gpu_ids = cfg['training']['device']['ids']

        self.logger = SummaryWriter(self.log_dir)

        if self.enable_log:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

    def prepare(self, cfg, resume):
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

        dataset = get_dataset(cfg, dataset_root, train_transform, valid_transform)
        criterion = get_criterion(cfg, dataset)
        net = get_net(cfg, dataset)
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

        if not resume:
            if self.enable_log:
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

        if self.enable_log:
            file = os.path.join(log_dir, 'cfg.yml')
            with open(file, 'w', encoding='utf-8') as fp:
                yaml.dump(cfg, fp, default_flow_style=False, Dumper=yaml.RoundTripDumper)

        self.net = net
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.log_dir = log_dir
        self.start_epoch = start_epoch

    def run(self):
        if self.enable_log:
            self.log_info()

        print('{:-^40s}'.format(' Start training '))
        self.show_train_info()

        if self.cuda:
            if self.distributed:
                self.net = apex.parallel.convert_syncbn_model(self.net).cuda()
                self.net = DistributedDataParallel(self.net)
            else:
                self.net = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids).cuda()

            self.criterion = self.criterion.cuda()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        valid_acc = 0.0
        best_acc = 0.0
        best_epoch = 0

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
                loss = self.training(epoch)
                if self.enable_log:
                    self.logger.add_scalar('loss', loss, epoch)

                if self.enable_log and (epoch + 1) % self.eval_epoch_interval == 0:
                    self.net.eval()
                    torch.set_grad_enabled(False)

                    # Evaluation phase
                    if self.eval_method == 'image' or self.distributed:
                        train_acc = self.evaluation(epoch, 'train')
                    else:
                        train_acc = self.evaluation_by_volume(epoch, 'train')

                    # Validation phase
                    if self.eval_method == 'image' or self.distributed:
                        valid_acc = self.evaluation(epoch, 'valid')
                    else:
                        valid_acc = self.evaluation_by_volume(epoch, 'valid')

                    print('Train data {} acc:  {:.5f}'.format(self.eval_func, train_acc))
                    print('Valid data {} acc:  {:.5f}'.format(self.eval_func, valid_acc))

            except KeyboardInterrupt:
                if self.enable_log:
                    cp_path = os.path.join(self.checkpoint_dir, 'INTERRUPTED.pth')
                    cp.save(epoch, self.net.module, self.optimizer, cp_path)
                return

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch
                checkpoint_filename = 'best.pth'
                cp.save(epoch, self.net, self.optimizer, os.path.join(self.checkpoint_dir, checkpoint_filename))
                print('Update best acc!')
                self.logger.add_scalar('best epoch', best_epoch + 1, 0)
                self.logger.add_scalar('best acc', best_acc, 0)

            if (epoch + 1) % self.checkpoint_epoch_interval == 0:
                checkpoint_filename = 'cp_{:03d}.pth'.format(epoch + 1)
                cp.save(epoch, self.net.module, self.optimizer, os.path.join(self.checkpoint_dir, checkpoint_filename))

            print(f'Best epoch: {best_epoch + 1}')
            print(f'Best acc: {best_acc:.5f}')

    def show_train_info(self):
        sampler = SequentialSampler(self.dataset.train_dataset)
        train_loader = DataLoader(self.dataset.train_dataset, batch_size=self.batch_size, sampler=sampler,
                                  num_workers=self.num_workers, pin_memory=True)

        sampler = SequentialSampler(self.dataset.valid_dataset)
        valid_loader = DataLoader(self.dataset.valid_dataset, batch_size=self.batch_size, sampler=sampler,
                                  num_workers=self.num_workers, pin_memory=True)

        msg = 'Net: {}\n'.format(self.net.__class__.__name__) + \
              'Dataset: {}\n'.format(self.dataset.__class__.__name__) + \
              'Epochs: {}\n'.format(self.epoch_num) + \
              'Learning rate: {}\n'.format(self.optimizer.param_groups[0]['lr']) + \
              'Batch size: {}\n'.format(self.batch_size) + \
              'Training size: {}\n'.format(len(train_loader.sampler)) + \
              'Validation size: {}\n'.format(len(valid_loader.sampler))

        if not self.distributed:
            if self.cuda:
                device = 'cuda' + str(self.gpu_ids)
            else:
                device = 'cpu'
            msg += 'Device: {}\n'.format(device)
        else:
            msg += 'Distributed size: {}\n'.format(torch.distributed.get_world_size())

        print(msg)

        if self.enable_log:
            self.logger.add_text('detail', msg)

    def log_info(self):
        net_summary_text = net_summary(self.net, self.dataset, device='cpu')
        self.logger.add_text('summary', net_summary_text)
        with open(os.path.join(self.log_dir, 'summary.txt'), 'w') as file:
            file.write(net_summary_text)
        if self.print_summary:
            print(net_summary_text)

        try:
            dummy_input = torch.zeros_like(self.dataset.__getitem__(0)[0])
            dummy_input = dummy_input.view((1,) + dummy_input.shape)
            self.logger.add_graph(self.net, dummy_input)
        except RuntimeError as e:
            # https://github.com/pytorch/pytorch/issues/10942#issuecomment-481992493
            # the forward function in the deeplab aspp has F.interpolate, kernel is dynamic,
            # but ONNX is statically determine the kernel size, so deeplab cannot export to ONNX currently.
            print(type(e), str(e))
            print('Warning: Cannot export net to ONNX, ignore log graph to tensorboard')

        sampler = SequentialSampler(self.dataset.valid_dataset)
        valid_loader = DataLoader(self.dataset.valid_dataset, batch_size=self.batch_size, sampler=sampler,
                                  num_workers=self.num_workers, pin_memory=True)
        for batch_idx, (imgs, labels, idx) in enumerate(valid_loader):
            imgs = make_grid(imgs)
            labels = make_grid(labels.unsqueeze(dim=1))
            _, labels, _ = self.dataset.vis_transform(labels=labels)
            self.logger.add_images('image', imgs, 0, dataformats='CHW')
            self.logger.add_images('label', labels, 0, dataformats='NCHW')
            break

    def training(self, epoch):
        if self.distributed:
            sampler = DistributedSampler(self.dataset.train_dataset)
            sampler.set_epoch(epoch)
        else:
            sampler = RandomSampler(self.dataset.train_dataset)

        train_loader = DataLoader(self.dataset.train_dataset, batch_size=self.batch_size, sampler=sampler,
                                  num_workers=self.num_workers, pin_memory=True)

        tbar = tqdm(train_loader, ascii=True, desc='train', dynamic_ncols=True)
        for batch_idx, (imgs, labels, idx) in enumerate(tbar):
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

    def evaluation(self, epoch, type):
        type = type.lower()
        if type == 'train':
            subset = self.dataset.train_dataset
        elif type == 'valid':
            subset = self.dataset.valid_dataset
        elif type == 'test':
            subset = self.dataset.test_dataset

        sampler = SequentialSampler(subset)
        data_loader = DataLoader(subset, batch_size=self.batch_size, sampler=sampler,
                                 num_workers=self.num_workers, pin_memory=True)

        evaluator = Evaluator(self.dataset.num_classes)
        tbar = tqdm(data_loader, desc=f'eval/{type:5}', ascii=True, dynamic_ncols=True)
        for batch_idx, (imgs, labels, idx) in enumerate(tbar):
            if self.cuda:
                imgs = imgs.cuda()

            outputs = self.net(imgs).argmax(dim=1)

            np_outputs = outputs.cpu().detach().numpy()
            np_labels = labels.cpu().detach().numpy()
            evaluator.add_batch(np_outputs, np_labels)

            if self.enable_log and type == 'valid' and \
                    self.visualize_iter_interval > 0 and batch_idx % self.visualize_iter_interval == 0:
                vis_imgs, vis_labels, vis_outputs = self.dataset.vis_transform(imgs, labels, outputs)
                imshow(title='Valid', imgs=(vis_imgs[0], vis_labels[0], vis_outputs[0]), shape=(1, 3),
                       subtitle=('image', 'label', 'predict'))
                self.logger.add_images('output', vis_outputs, epoch, dataformats='NCHW')

        acc = evaluator.eval(self.eval_func)

        if self.enable_log:
            evaluator.log_acc(self.logger, epoch, f'{type}/')
        return acc

    def evaluation_by_volume(self, epoch, type):
        type = type.lower()
        if type == 'train':
            subset = self.dataset.train_dataset
            case = self.dataset.case_indices[self.dataset.split_case:]
            case.pop(0)
        elif type == 'valid':
            subset = self.dataset.valid_dataset
            case = self.dataset.case_indices[:self.dataset.split_case]
            vis_idx = [i + j for i, j in zip(case, self.dataset.vis_idx)]

        vol_case_i = 0
        vol_label = []
        vol_output = []

        vis_case_i = 0
        vis_img = []
        vis_label = []
        vis_output = []

        sampler = SequentialSampler(subset)
        data_loader = DataLoader(subset, batch_size=self.batch_size, sampler=sampler,
                                 num_workers=self.num_workers, pin_memory=True)

        evaluator = Evaluator(self.dataset.num_classes)

        with tqdm(total=len(case), ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
            for batch_idx, (imgs, labels, idx) in enumerate(data_loader):
                if self.cuda:
                    imgs = imgs.cuda()
                outputs = self.net(imgs).argmax(dim=1)

                np_labels = labels.cpu().detach().numpy()
                np_outputs = outputs.cpu().detach().numpy()
                idx = idx.numpy()

                vol_label.append(np_labels)
                vol_output.append(np_outputs)

                while type == 'valid' and vis_case_i < len(vis_idx) and idx[-1] >= vis_idx[vis_case_i]:
                    img_idx = len(imgs) - (idx[-1] - vis_idx[vis_case_i]) - 1
                    vis_img.append(imgs[img_idx].unsqueeze(0).cpu().detach())
                    vis_label.append(labels[img_idx].unsqueeze(0).cpu().detach())
                    vis_output.append(outputs[img_idx].unsqueeze(0).cpu().detach())
                    vis_case_i += 1

                while vol_case_i < len(case) and idx[-1] >= case[vol_case_i] - 1:
                    vol_output = np.concatenate(vol_output, axis=0)
                    vol_label = np.concatenate(vol_label, axis=0)

                    vol_idx = case[vol_case_i] - case[vol_case_i - 1]
                    evaluator.add(vol_output[:vol_idx], vol_label[:vol_idx])

                    vol_output = [vol_output[vol_idx:]]
                    vol_label = [vol_label[vol_idx:]]
                    vol_case_i += 1
                    pbar.update(1)

        if type == 'valid':
            vis_img = torch.cat(vis_img, dim=0).numpy()
            vis_label = torch.cat(vis_label, dim=0).numpy()
            vis_output = torch.cat(vis_output, dim=0).numpy()
            vis_img, vis_label, vis_output = self.dataset.vis_transform(vis_img, vis_label, vis_output)
            self.logger.add_images('image', vis_img, epoch)
            self.logger.add_images('label', vis_label, epoch)
            self.logger.add_images('output', vis_output, epoch)

        acc = evaluator.eval(self.eval_func)
        evaluator.log_acc(self.logger, epoch, f'{type}/')
        return acc


def get_args():
    parser = ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument('-c', '--config', type=str, default='./configs/unet_spineseg.yml',
                        help='load config file')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='resume checkpoint')
    parser.add_argument('-d', '--distributed', default=False, type=bool)
    parser.add_argument('-l', '--local_rank', default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    enable_log = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.backends.cudnn.benchmark = True
        if dist.get_rank() != 0:
            enable_log = False
            sys.stdout = open(os.devnull, 'w')

    if not args.resume:
        cfg_file = args.config
    else:
        cp_file = pathlib.Path(args.resume).expanduser()
        cfg_file = cp_file.parents[1] / 'cfg.yml'
        assert cp_file.is_file() and cfg_file.is_file()

    cfg_file = os.path.expanduser(cfg_file)
    with open(cfg_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.RoundTripLoader)

    torch.cuda.empty_cache()
    trainer = Trainer(cfg, args.resume, args.distributed, enable_log)
    trainer.run()
