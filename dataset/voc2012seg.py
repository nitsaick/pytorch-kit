import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data

import dataset.transform as transform


class VOC2012Seg(data.Dataset):
    def __init__(self, root, train_transform=None, valid_transform=None):
        
        self.imgs, self.labels = self.get_img_list(root)
        
        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()
        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        
        self._split()
        
        self.img_channels = self.__getitem__(0)[0].shape[0]
    
    def get_img_list(self, root):
        list_root = os.path.join(root, 'ImageSets', 'Segmentation')
        img_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'SegmentationClass')
        
        train_list = os.path.join(list_root, 'train.txt')
        with open(train_list) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        
        imgs = [os.path.join(img_root, name + '.jpg') for name in content]
        labels = [os.path.join(label_root, name + '.png') for name in content]
        
        split = len(imgs)
        
        val_list = os.path.join(list_root, 'val.txt')
        with open(val_list) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        
        imgs += [os.path.join(img_root, name + '.jpg') for name in content]
        labels += [os.path.join(label_root, name + '.png') for name in content]
        
        self.indices = list(range(len(imgs)))
        self.train_indices, self.valid_indices = self.indices[:split], self.indices[split:]
        
        return imgs, labels
    
    def _split(self):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        
        self.train_sampler = data.RandomSampler(self.train_dataset)
        self.valid_sampler = data.SequentialSampler(self.valid_dataset)
        self.test_sampler = data.SequentialSampler(self)
    
    def get_dataloader(self, batch_size=1, num_workers=0):
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size,
                                       sampler=self.train_sampler, num_workers=num_workers)
        valid_loader = data.DataLoader(self.valid_dataset, batch_size=batch_size,
                                       sampler=self.valid_sampler, num_workers=num_workers)
        test_loader = data.DataLoader(self, batch_size=batch_size, sampler=self.test_sampler, num_workers=num_workers)
        return train_loader, valid_loader, test_loader
    
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
    
    def default_transform(self, data):
        data = transform.to_tensor(data)
        return data
    
    def get_classes_name(self):
        classes_name = ['background', 'aeroplane', 'bicycle',
                        'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat',
                        'chair', 'cow', 'dining table',
                        'dog', 'horse', 'motorbike',
                        'person', 'potted plant', 'sheep',
                        'sofa', 'train', 'tv/monitor']
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
        
        label = (label * 255).long().view(label.shape[1], label.shape[2])
        
        return img, label
    
    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    from utils.vis import imshow
    from dataset.transform import *
    
    root = os.path.expanduser('~/dataset/VOC2012Seg')
    dataset_ = VOC2012Seg(root=root, train_transform=real_world_transform(output_size=256, type='train'),
                          valid_transform=real_world_transform(output_size=256, type='valid'))
    
    train_loader, _, _ = dataset_.get_dataloader(batch_size=1)
    for batch_idx, (img, label) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None, to_plt=True)
        imshow(title='VOC2012Seg', imgs=(imgs[0], labels[0]))
        break
