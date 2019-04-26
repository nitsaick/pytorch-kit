import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data

import dataset.transform as transform
from utils.func import recursive_glob


class SpineSeg(data.Dataset):
    def __init__(self, root, valid_rate=0.2, train_transform=None, valid_transform=None):
        self.root = root
        self.imgs, self.labels = self.get_img_list(root)

        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()

        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self._split(valid_rate)

        self.img_channels = self.__getitem__(0)[0].shape[0]

    def _split(self, valid_rate):
        self.indices = list(range(self.dataset_size))
        split = int(np.floor(valid_rate * self.dataset_size))
        self.train_indices, self.valid_indices = self.indices[split:], self.indices[:split]
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        self.test_dataset = self

    def get_img_list(self, root):
        img_root = os.path.join(root, 'image')
        label_root = os.path.join(root, 'label')

        imgs = sorted(recursive_glob(root=img_root, suffix='jpg'))
        labels = sorted(recursive_glob(root=label_root, suffix='jpg'))

        return imgs, labels

    def default_transform(self, data):
        data = transform.to_tensor(data)
        image, label = data['image'], data['label']

        image = image.view(1, *image.shape).float() / 255
        label = torch.where(label > 128, torch.ones_like(label), torch.zeros_like(label)).long()

        return {'image': image, 'label': label}

    def get_classes_name(self):
        classes_name = ['background', 'target']
        return classes_name

    def get_colormap(self):
        cmap = [[0, 0, 0], [255, 255, 255]]
        cmap = np.array(cmap, dtype=np.int)
        return cmap

    def vis_transform(self, imgs=None, labels=None, preds=None, to_plt=False):
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
        img = Image.open(img_path)

        label_path = self.labels[index]
        label = Image.open(label_path)

        data = {'image': img, 'label': label}
        if index in self.train_indices and self.train_transform is not None:
            data = self.train_transform(data)
        elif index in self.valid_indices and self.valid_transform is not None:
            data = self.valid_transform(data)

        data = self.default_transform(data)

        img = data['image']
        label = data['label']

        return img, label, index

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    from utils.vis import imshow
    from dataset.transform import *

    root = os.path.expanduser('~/dataset/SpineSeg')
    dataset_ = SpineSeg(root=root, valid_rate=0.2,
                        train_transform=medical_transform(output_size=(128, 256), scale_range=0, type='train'),
                        valid_transform=None)

    train_loader, _, _ = dataset_.get_dataloader(batch_size=1)

    for batch_idx, (img, label) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None)
        imshow(title='SpineSeg', imgs=(imgs[0], labels[0]))
        break
