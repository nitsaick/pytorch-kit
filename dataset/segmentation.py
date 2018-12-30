import os
import pickle
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torchvision.transforms.functional
from PIL import Image
from torch.utils import data

import dataset.transform as transform


class BaseSegDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, root, valid_rate=0.2, transform=None, transform_params=None,
                 shuffle=False, resume=False, log_dir=None):
        self.imgs, self.labels = self.get_img_list(root)
        
        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()
        
        self.transform = transform
        self.transform_params = transform_params
        
        if log_dir is None:
            log_dir = root
        
        indices_filename = os.path.join(log_dir, 'indices.npy')
        if resume:
            with open(indices_filename, 'rb') as fp:
                self.indices = pickle.load(fp)
        else:
            if os.path.exists(log_dir) is False:
                os.makedirs(log_dir)
            self.indices = list(range(self.dataset_size))
            if shuffle:
                np.random.shuffle(self.indices)
            with open(indices_filename, 'wb') as fp:
                pickle.dump(self.indices, fp)
        
        split = int(np.floor(valid_rate * self.dataset_size))
        self.train_indices, self.valid_indices = self.indices[split:], self.indices[:split]
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        
        self.train_sampler = data.RandomSampler(self.train_dataset)
        self.valid_sampler = data.SequentialSampler(self.valid_dataset)
        self.all_sampler = data.SequentialSampler(self)
        
        self.img_channels = self.__getitem__(0)[0].shape[0]
    
    def get_dataloader(self, batch_size=1):
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.train_sampler)
        valid_loader = data.DataLoader(self.valid_dataset, batch_size=batch_size, sampler=self.valid_sampler)
        all_loader = data.DataLoader(self, batch_size=batch_size, sampler=self.all_sampler)
        return train_loader, valid_loader, all_loader
    
    @abstractmethod
    def get_img_list(self, root):
        pass
    
    @abstractmethod
    def get_classes_name(self):
        pass
    
    @abstractmethod
    def get_colormap(self):
        pass
    
    @abstractmethod
    def default_transform(self, img, label):
        pass
    
    def vis_transform(self, imgs, labels, preds, to_plt=False):
        cmap = self.get_colormap()
        if imgs is not None:
            if type(imgs).__module__ != np.__name__:
                imgs = imgs.cpu().detach().numpy()
            if to_plt is True:
                imgs = imgs.transpose((0, 2, 3, 1))
        
        if labels is not None:
            if type(labels).__module__ != np.__name__:
                labels = labels.cpu().detach().numpy()
            labels = cmap[labels]
            labels = labels.transpose((0, 3, 1, 2))
            if to_plt is True:
                labels = labels.transpose((0, 2, 3, 1))
        
        if preds is not None:
            if type(preds).__module__ != np.__name__:
                preds = preds.cpu().detach().numpy()
            if preds.shape[1] == self.num_classes:
                preds = preds.argmax(axis=1)
            preds = cmap[preds]
            preds = preds.transpose((0, 3, 1, 2))
            if to_plt is True:
                preds = preds.transpose((0, 2, 3, 1))
        
        return imgs, labels, preds
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        
        label_path = self.labels[index]
        label = Image.open(label_path)
        
        if index in self.train_indices and self.transform is not None:
            img, label = self.transform(img, label, self.transform_params)
        
        img, label = self.default_transform(img, label)
        
        return img, label
    
    def __len__(self):
        return self.dataset_size


class SpineSeg(BaseSegDataset):
    def __init__(self, root, valid_rate=0.2, transform=None, transform_params=None,
                 shuffle=False, resume=False, log_dir=None):
        super(SpineSeg, self).__init__(root, valid_rate, transform, transform_params, shuffle, resume, log_dir)
    
    def get_img_list(self, root):
        img_root = os.path.join(root, 'image')
        label_root = os.path.join(root, 'label')
        
        imgs = [os.path.join(img_root, img) for img in sorted(os.listdir(img_root))]
        labels = [os.path.join(label_root, img) for img in sorted(os.listdir(label_root))]
        
        return imgs, labels
    
    def default_transform(self, img, label):
        img, label = transform.to_tensor(img, label)
        label = torch.where(label > 0.5, torch.ones_like(label), torch.zeros_like(label))
        label = label.long().view(label.shape[1], label.shape[2])
        return img, label
    
    def get_classes_name(self):
        classes_name = ['background', 'target']
        return classes_name
    
    def get_colormap(self):
        cmap = [[0, ], [255, ]]
        cmap = np.array(cmap, dtype=np.int)
        return cmap


class xVertSeg(BaseSegDataset):
    def __init__(self, root, valid_rate=0.2, transform=None, transform_params=None,
                 shuffle=False, resume=False, log_dir=None):
        super(xVertSeg, self).__init__(root, valid_rate, transform, transform_params, shuffle, resume, log_dir)
    
    def get_img_list(self, root):
        dirs = [dir_ for dir_ in os.listdir(root) if os.path.isdir(os.path.join(root, dir_))]
        
        imgs = []
        labels = []
        
        for dir_ in dirs:
            dir_ = os.path.join(root, dir_)
            
            img_root = os.path.join(dir_, 'image')
            label_root = os.path.join(dir_, 'label')
            
            imgs += [os.path.join(img_root, img) for img in sorted(os.listdir(img_root))]
            labels += [os.path.join(label_root, img) for img in sorted(os.listdir(label_root))]
        
        return imgs, labels
    
    def default_transform(self, img, label):
        img, label = transform.to_tensor(img, label)
        label = torch.where(label > 0.5, torch.ones_like(label), torch.zeros_like(label))
        label = label.long().view(label.shape[1], label.shape[2])
        return img, label
    
    def get_classes_name(self):
        classes_name = ['background', 'target']
        return classes_name
    
    def get_colormap(self):
        cmap = [[0, ], [255, ]]
        cmap = np.array(cmap, dtype=np.int)
        return cmap


class VOC2012Seg(BaseSegDataset):
    def __init__(self, root, valid_rate=0.2, transform=None, transform_params=None,
                 shuffle=False, resume=False, log_dir=None, resize=None):
        self.cmap = self.get_colormap()
        self.resize = resize
        super(VOC2012Seg, self).__init__(root, valid_rate, transform, transform_params, shuffle, resume, log_dir)
    
    def get_img_list(self, root):
        img_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'SegmentationClass')
        
        imgs = [os.path.join(img_root, img.split('.')[0] + '.jpg') for img in sorted(os.listdir(label_root))]
        labels = [os.path.join(label_root, img.split('.')[0] + '.png') for img in sorted(os.listdir(label_root))]
        
        return imgs, labels
    
    def get_colormap(self, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)
        
        dtype = np.float32 if normalized else np.uint8
        cmap_range = 256
        cmap = np.zeros((cmap_range, 3), dtype=dtype)
        for i in range(cmap_range):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3
            
            cmap[i] = np.array([r, g, b])
        
        cmap = cmap / 255 if normalized else cmap
        return cmap
    
    def default_transform(self, img, label):
        if self.resize:
            img = torchvision.transforms.functional.resize(img, self.resize)
            label = torchvision.transforms.functional.resize(label, self.resize)
        img, label = transform.to_tensor(img, label)
        label = (label * 255).long().view(label.shape[1], label.shape[2])
        
        return img, label
    
    def get_classes_name(self):
        classes_name = ['background', 'aeroplane', 'bicycle',
                        'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat',
                        'chair', 'cow', 'dining table',
                        'dog', 'horse', 'motorbike',
                        'person', 'potted plant', 'sheep',
                        'sofa', 'train', 'tv/monitor']
        return classes_name


if __name__ == '__main__':
    from utils.vis import imshow
    from dataset.transform import *
    
    root = os.path.expanduser('~/dataset/SpineSeg')
    dataset_ = SpineSeg(root=root, shuffle=True, valid_rate=0.2, transform=random_flip_transform)
    
    train_loader, _, _ = dataset_.get_dataloader(batch_size=1)
    
    for batch_idx, (img, label) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None)
        imshow(title='SpineSeg', imgs=(imgs[0], labels[0]))
        break
    
    root = os.path.expanduser('~/dataset/xVertSeg')
    dataset_ = xVertSeg(root=root, shuffle=True, valid_rate=0.2, transform=random_flip_transform)
    
    train_loader, _, _ = dataset_.get_dataloader(batch_size=1)
    
    for batch_idx, (img, label) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None)
        imshow(title='xVertSeg', imgs=(imgs[0], labels[0]))
        break
    
    root = os.path.expanduser('~/dataset/VOC2012Seg')
    dataset_ = VOC2012Seg(root=root, shuffle=True, valid_rate=0.2,
                          transform=random_crop_transform, transform_params=(128, 256))
    
    train_loader, _, _ = dataset_.get_dataloader(batch_size=1)
    for batch_idx, (img, label) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None, to_plt=True)
        imshow(title='VOC2012Seg', imgs=(imgs[0], labels[0]))
        break
