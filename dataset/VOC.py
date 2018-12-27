import os
import pickle

import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils import data

import utils.visualize as visualize


class VOC2012(data.Dataset):
    def __init__(self, root, transform=None, resume=False, shuffle=False, valid_rate=0.2, log_dir='./'):
        self.class_name = self.get_class_name()
        self.num_class = len(self.class_name)
        self.img_channels = 3
        self.cmap = self.get_colormap()

        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'SegmentationClass')

        self.images = [os.path.join(image_root, img.split('.')[0] + '.jpg') for img in os.listdir(label_root)]
        self.labels = [os.path.join(label_root, img.split('.')[0] + '.png') for img in os.listdir(label_root)]

        if transform is None:
            self.transform = self.default_transform
        else:
            self.transform = transform

        dataset_size = len(self.images)

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

        self.indices = list(range(len(self.images)))

        split = int(np.floor(valid_rate * dataset_size))
        self.train_indices, self.valid_indices = self.indices[split:], self.indices[:split]

        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)

        self.train_sampler = data.RandomSampler(self.train_dataset)
        self.valid_sampler = data.SequentialSampler(self.valid_dataset)

    def default_transform(self, image, label):
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        return image, label

    def get_dataloader(self, batch_size=1):
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.train_sampler)
        valid_loader = data.DataLoader(self.valid_dataset, batch_size=batch_size, sampler=self.valid_sampler)
        return train_loader, valid_loader

    def get_colormap(self, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
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

    def get_class_name(self):
        class_name = ['background', 'aeroplane', 'bicycle',
                      'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat',
                      'chair', 'cow', 'dining table',
                      'dog', 'horse', 'motorbike',
                      'person', 'potted plant', 'sheep',
                      'sofa', 'train', 'tv/monitor']
        return class_name

    def batch_visualize_transform(self, img, img_type, to_plt=False):
        if type(img).__module__ != np.__name__:
            img = img.cpu().detach().numpy()

        if img_type == 'image':
            img = img
        elif img_type == 'label':
            img = self.cmap[img]
            img = img.transpose((0, 3, 1, 2))
        elif img_type == 'predict':
            if img.shape[1] == self.num_class:
                img = img.argmax(axis=1)
            img = self.cmap[img.astype(int)]
            img = img.transpose((0, 3, 1, 2))

        if to_plt is True:
            img = img.transpose((0, 2, 3, 1))

        return img

    def label_to_multiclass(self, label):
        class_capture = np.zeros((self.num_class, 256))

        label_data = np.array(label)
        label_multiclass = np.zeros((self.num_class,) + label_data.shape, dtype=np.uint8)

        for i in range(self.num_class):
            class_capture[i][i] = 255
            label_multiclass[i] = class_capture[i][label_data]

        label_multiclass = label_multiclass.transpose((1, 2, 0))

        return label_multiclass

    def __getitem__(self, index):
        image_path = self.images[index]
        img = Image.open(image_path)

        label_path = self.labels[index]
        label = Image.open(label_path)

        resize = (480, 320)
        img = img.resize(resize, Image.BILINEAR)
        label = label.resize(resize, Image.BILINEAR)

        if (index in self.train_indices):
            img, label = self.transform(img, label)
        else:
            img, label = self.default_transform(img, label)

        label = label * 255
        label = label.long().view(320, 480)
        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = VOC2012(root='./dataset/VOC2012/')

    batch_size = 1
    train_loader, _ = dataset.get_dataloader(batch_size)

    num_epochs = 1

    fig = visualize.create_fig((1, 2))
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        for batch_index, (img, label) in enumerate(train_loader):
            print('Batch Index:', batch_index)
            a = dataset.batch_visualize_transform(img, 'image', to_plt=True)
            b = dataset.batch_visualize_transform(label, 'label', to_plt=True)
            visualize.imshow((a[0], b[0]), shape=(1, 2), fig=fig)
