import os

import numpy as np
from PIL import Image
from torch.utils import data

import dataset.transform as transform
from utils.func import recursive_glob


class Cityscapes(data.Dataset):
    def __init__(self, root, train_transform=None, valid_transform=None):
        self.root = root
        self.imgs, self.labels = self.get_img_list(root)

        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()

        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

        self._split()

        self.img_channels = self.__getitem__(0)[0].shape[0]

    def get_img_list(self, root):
        img_root = os.path.join(root, 'leftImg8bit')
        label_root = os.path.join(root, 'gtFine')

        imgs = []
        labels = []

        train_img_root = os.path.join(img_root, 'train')
        train_label_root = os.path.join(label_root, 'train')
        imgs += sorted(recursive_glob(root=train_img_root, suffix='leftImg8bit.png'))
        labels += sorted(recursive_glob(root=train_label_root, suffix='gtFine_labelIds.png'))

        train_split = len(imgs)

        valid_img_root = os.path.join(img_root, 'val')
        valid_label_root = os.path.join(label_root, 'val')
        imgs += sorted(recursive_glob(root=valid_img_root, suffix='leftImg8bit.png'))
        labels += sorted(recursive_glob(root=valid_label_root, suffix='gtFine_labelIds.png'))

        valid_split = len(imgs)

        test_img_root = os.path.join(img_root, 'test')
        test_label_root = os.path.join(label_root, 'test')
        imgs += sorted(recursive_glob(root=test_img_root, suffix='leftImg8bit.png'))
        labels += sorted(recursive_glob(root=test_label_root, suffix='gtFine_labelIds.png'))

        self.indices = list(range(len(imgs)))
        self.train_indices = self.indices[:train_split]
        self.valid_indices = self.indices[train_split:valid_split]
        self.test_indices = self.indices[valid_split:]

        return imgs, labels

    def _split(self):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        self.test_dataset = data.Subset(self, self.test_indices)

    def get_colormap(self):
        cmap = np.array([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]]
        ).astype(np.uint8)

        other_map = np.array([[0, 0, 0], ] * (256 - cmap.shape[0]))
        cmap = np.concatenate((cmap, other_map))
        return cmap

    def default_transform(self, data):
        data = transform.to_tensor(data)
        image, label = data['image'], data['label']

        image = image.permute(2, 0, 1).float() / 255
        label = label.long()

        return {'image': image, 'label': label}

    def get_classes_name(self):
        classes_name = ['road', 'sidewalk', 'building',
                        'wall', 'fence', 'pole',
                        'traffic_light', 'traffic_sign', 'vegetation',
                        'terrain', 'sky', 'person',
                        'rider', 'car', 'truck',
                        'bus', 'train', 'motorcycle',
                        'bicycle']
        return classes_name

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
        else:
            data = transform.to_numpy(data)

        img = data['image']
        label = data['label']

        for _voidc in self.void_classes:
            label[label == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            label[label == _validc] = self.class_map[_validc]

        data = {'image': img, 'label': label}
        data = self.default_transform(data)

        img = data['image']
        label = data['label']

        return img, label, index

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    from utils.vis import imshow
    from dataset.transform import *

    root = os.path.expanduser('~/dataset/Cityscapes')
    dataset_ = Cityscapes(root=root,
                          train_transform=real_world_transform(output_size=512, scale_range=0.2, type='train'),
                          valid_transform=real_world_transform(output_size=512, scale_range=0.2, type='valid'))

    train_loader, _, _ = dataset_.get_dataloader(batch_size=1)
    for batch_idx, (img, label) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None, to_plt=True)
        imshow(title='Cityscapes', imgs=(imgs[0], labels[0]))
        break
