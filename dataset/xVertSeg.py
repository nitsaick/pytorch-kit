import os
import pickle

import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils import data

from utils.visualize import imshow


class xVertSeg(data.Dataset):
    def __init__(self, root, transform=None, resume=False, shuffle=False, log_dir=None, valid_rate=0.2):
        self.num_class = 1
        self.img_channels = 1

        dirs = os.listdir(root)

        imgs = []
        labels = []

        for dir in dirs:
            dir = os.path.join(root, dir)

            img_root = os.path.join(dir, 'image')
            label_root = os.path.join(dir, 'label')

            imgs += [os.path.join(img_root, img) for img in os.listdir(img_root)]
            labels += [os.path.join(label_root, img) for img in os.listdir(label_root)]

        if len(imgs) != len(labels):
            raise AssertionError('image和label的數量不一致')

        self.imgs = imgs
        self.labels = labels

        if transform is None:
            self.transform = self.default_transform
        else:
            self.transform = transform

        dataset_size = len(imgs)

        if log_dir is None:
            log_dir = root

        indices_filename = os.path.join(log_dir, 'indices.npy')
        if resume:
            with open(indices_filename, 'rb') as fp:  # Unpickling
                self.indices = pickle.load(fp)
        else:
            if os.path.exists(log_dir) is False:
                os.makedirs(log_dir)
            self.indices = list(range(dataset_size))
            if shuffle:
                np.random.shuffle(self.indices)
            with open(indices_filename, 'wb') as fp:  # Pickling
                pickle.dump(self.indices, fp)

        split = int(np.floor(valid_rate * dataset_size))
        self.train_indices, self.valid_indices = self.indices[split:], self.indices[:split]
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)

        self.train_sampler = data.RandomSampler(self.train_dataset)
        self.valid_sampler = data.SequentialSampler(self.valid_dataset)

        self.test_sampler = data.SequentialSampler(self)

    def batch_visualize_transform(self, img, label, pred, to_plt=False):
        if img is not None:
            if type(img).__module__ != np.__name__:
                img = img.cpu().detach().numpy()
            if to_plt is True:
                img = img.transpose((0, 2, 3, 1))

        if label is not None:
            if type(label).__module__ != np.__name__:
                label = label.cpu().detach().numpy()
            if to_plt is True:
                label = label.transpose((0, 2, 3, 1))

        if pred is not None:
            if type(pred).__module__ != np.__name__:
                pred = pred.cpu().detach().numpy()
            if to_plt is True:
                pred = pred.transpose((0, 2, 3, 1))

        return img, label, pred

    def default_transform(self, image, label):
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        return image, label

    def get_dataloader(self, batch_size=1):
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size,
                                       sampler=self.train_sampler)
        valid_loader = data.DataLoader(self.valid_dataset, batch_size=batch_size,
                                       sampler=self.valid_sampler)
        test_loader = data.DataLoader(self, batch_size=batch_size,
                                      sampler=self.test_sampler)
        return train_loader, valid_loader, test_loader

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)

        label_path = self.labels[index]
        label = Image.open(label_path)

        if (index in self.train_indices):
            img, label = self.transform(img, label)
        else:
            img, label = self.default_transform(img, label)

        return img, label  # 可以改為輸出numpy，DataLoader也會自動轉為Tensor，但不會做標準化，如果使用PIL讀圖，需手動轉換成Tensor

    def __len__(self):
        return len(self.imgs)


# Example
if __name__ == '__main__':
    dataset = xVertSeg(root='./dataset/xVertSeg', shuffle=False, valid_rate=0.2)

    train_loader, _, _ = dataset.get_dataloader(batch_size=1)

    num_epochs = 1
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        for batch_index, (img, label) in enumerate(train_loader):
            print('Batch Index:', batch_index)
            imshow(main_title='SpineSeg', imgs=(img[0][0], label[0][0]), cmap='gray')
