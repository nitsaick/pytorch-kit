import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils import data
from tqdm import trange

from utils.metrics import Evaluator


class kits19(data.Dataset):
    def __init__(self, root, valid_rate=0.2, train_transform=None, valid_transform=None, distributed=False,
                 conversion=False):
        self.root = Path(root)
        if conversion:
            self.conversion_nii2npy(self.root)
        self.imgs, self.labels = self.get_img_list(self.root, valid_rate)

        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()

        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self._split(distributed)

        self.img_channels = self.__getitem__(0)[0].shape[0]

    def _split(self, distributed):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)

        if distributed:
            self.train_sampler = data.distributed.DistributedSampler(self.train_dataset)
        else:
            self.train_sampler = data.RandomSampler(self.train_dataset)
        self.valid_sampler = data.SequentialSampler(self.valid_dataset)
        self.test_sampler = data.SequentialSampler(self)

    def get_dataloader(self, batch_size=1, num_workers=0, pin_memory=False):
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.train_sampler,
                                       num_workers=num_workers, pin_memory=pin_memory)
        valid_loader = data.DataLoader(self.valid_dataset, batch_size=batch_size, sampler=self.valid_sampler,
                                       num_workers=num_workers, pin_memory=pin_memory)
        test_loader = data.DataLoader(self, batch_size=batch_size, sampler=self.test_sampler,
                                      num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader, test_loader

    def eval(self, net, cuda, eval_func, logger, epoch, type, batch_size, num_workers, pin_memory):
        if type == 'eval':
            case = self.case_indices[self.split_case:]
        elif type == 'valid':
            case = self.case_indices[:self.split_case]

        vis_img = []
        vis_label = []
        vis_output = []
        vis_idx = [300, 264, 179, 188, 42, 333, 48, 40, 147, 45, 17, 19, 24, 40, 119, 41, 60, 37, 68, 42, 29, 25, 253,
                   43, 38, 63, 176, 366, 27, 76, 13, 41, 57, 172, 45, 40, 55, 32, 21, 37, 67]
        evaluator = Evaluator(self.num_classes)
        for case_i in trange(len(case) - 1, desc=f'{type:5}', ascii=True):
            vol_label = []
            vol_output = []

            img_idx = list(range(case[case_i], case[case_i + 1]))
            case_subset = data.Subset(self, img_idx)
            sampler = data.SequentialSampler(case_subset)
            data_loader = data.DataLoader(case_subset, batch_size=batch_size, sampler=sampler,
                                          num_workers=num_workers, pin_memory=pin_memory)

            # total_num = case[case_i + 1] - case[case_i]

            i = 0
            appended = False
            for batch_idx, (img, label) in enumerate(data_loader):
                if cuda:
                    img = img.cuda()

                output = net(img)

                vol_label.append(label)
                vol_output.append(output)
                if type == 'valid' and not appended and i + img.shape[0] > vis_idx[case_i]:
                    idx = vis_idx[case_i] - i
                    vis_img.append(img[idx].unsqueeze(0).cpu().detach())
                    vis_label.append(label[idx].unsqueeze(0).cpu().detach())
                    vis_output.append(output[idx].unsqueeze(0).cpu().detach())
                    appended = True

                i += img.shape[0]

            vol_output = torch.cat(vol_output, dim=0).argmax(dim=1).cpu().detach().numpy()
            vol_label = torch.cat(vol_label, dim=0).cpu().detach().numpy()
            evaluator.add(vol_output, vol_label)

        if type == 'valid':
            vis_img = torch.cat(vis_img, dim=0).numpy()
            vis_label = torch.cat(vis_label, dim=0).numpy()
            vis_output = torch.cat(vis_output, dim=0).argmax(dim=1).numpy()
            vis_img, vis_label, vis_output = self.vis_transform(vis_img, vis_label, vis_output)
            logger.add_images('image', vis_img, epoch)
            logger.add_images('label', vis_label, epoch)
            logger.add_images('output', vis_output, epoch)

        acc = evaluator.eval(eval_func)
        evaluator.log_acc(logger, epoch, type + '/')
        return acc

    def normalize(self, volume):
        DEFAULT_HU_MAX = 512
        DEFAULT_HU_MIN = -512
        volume = np.clip(volume, DEFAULT_HU_MIN, DEFAULT_HU_MAX)

        mxval = np.max(volume)
        mnval = np.min(volume)
        volume_norm = (volume - mnval) / max(mxval - mnval, 1e-3)

        return volume_norm

    def conversion_nii2npy(self, root):
        cases = sorted([d for d in root.iterdir() if d.is_dir()])
        for case in cases:
            print(case)
            vol = nib.load(str(case / 'imaging.nii.gz')).get_data()
            vol = self.normalize(vol)
            imaging_dir = case / 'imaging'
            if not imaging_dir.exists():
                imaging_dir.mkdir()
            if len(list(imaging_dir.glob('*.npy'))) != vol.shape[0]:
                for i in range(vol.shape[0]):
                    np.save(str(imaging_dir / f'{i:03}.npy'), vol[i])

            seg = nib.load(str(case / 'segmentation.nii.gz')).get_data()
            segmentation_dir = case / 'segmentation'
            if not segmentation_dir.exists():
                segmentation_dir.mkdir()
            if len(list(segmentation_dir.glob('*.npy'))) != seg.shape[0]:
                for i in range(seg.shape[0]):
                    np.save(str(segmentation_dir / f'{i:03}.npy'), seg[i])

    def get_img_list(self, root, valid_rate):
        imgs = []
        labels = []

        self.case_indices = [0, ]
        cases = sorted([d for d in root.iterdir() if d.is_dir()])
        self.split_case = int(len(cases) * valid_rate)
        for case in cases:
            if case.stem == 'case_00160': continue

            imaging_dir = case / 'imaging'
            imgs += sorted(list(imaging_dir.glob('*.npy')))
            segmentation_dir = case / 'segmentation'
            labels += sorted(list(segmentation_dir.glob('*.npy')))

            assert len(imgs) == len(labels)
            self.case_indices.append(len(imgs))
            if case.stem[-3:] == f'{self.split_case:03}':
                split = len(imgs)

        self.indices = list(range(len(imgs)))
        self.train_indices = self.indices[split:]
        self.valid_indices = self.indices[:split]

        return imgs, labels

    def default_transform(self, data):
        image, label = data['image'], data['label']

        image = image.astype(np.float32)
        image = np.stack((image, image, image), 0)
        label = label.astype(np.int64)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return {'image': image, 'label': label}

    def get_colormap(self):
        cmap = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
        cmap = np.array(cmap, dtype=np.int)
        return cmap

    def get_classes_name(self):
        classes_name = ['background', 'kidney', 'tumor']
        return classes_name

    def vis_transform(self, imgs, labels, preds, to_plt=False):
        cmap = self.get_colormap()
        if imgs is not None:
            if type(imgs).__module__ != np.__name__:
                imgs = imgs.cpu().detach().numpy()
            if to_plt is True:
                imgs = imgs.transpose((0, 2, 3, 1))

        if labels is not None:
            if type(labels).__module__ != np.__name__:
                labels = labels.cpu().detach().numpy().astype('int')
            labels = cmap[labels]
            labels = labels.transpose((0, 3, 1, 2))
            if to_plt is True:
                labels = labels.transpose((0, 2, 3, 1))
            labels = labels / 255.

        if preds is not None:
            if type(preds).__module__ != np.__name__:
                preds = preds.cpu().detach().numpy()
            if preds.shape[1] == self.num_classes:
                preds = preds.argmax(axis=1)
            preds = cmap[preds]
            preds = preds.transpose((0, 3, 1, 2))
            if to_plt is True:
                preds = preds.transpose((0, 2, 3, 1))
            preds = preds / 255.

        return imgs, labels, preds

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = np.load(str(img_path))

        label_path = self.labels[index]
        label = np.load(str(label_path))

        data = {'image': img, 'label': label}
        if index in self.train_indices and self.train_transform is not None:
            data = self.train_transform(data)
        elif index in self.valid_indices and self.valid_transform is not None:
            data = self.valid_transform(data)

        data = self.default_transform(data)

        img = data['image']
        label = data['label']

        return img, label

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    from utils.vis import imshow

    root = os.path.expanduser('D:/Qsync/kits19/data')
    dataset_ = kits19(root=root, valid_rate=0.2,
                      train_transform=None,
                      valid_transform=None)

    train_loader, valid_loader, _ = dataset_.get_dataloader(batch_size=1)
    dataset_.eval()
    for batch_idx, (img, label) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None)
        imshow(title='kits19', imgs=(imgs[0], labels[0]))
