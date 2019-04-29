import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils import data


class kits19(data.Dataset):
    def __init__(self, root, valid_rate=0.3, train_transform=None, valid_transform=None,
                 conversion=False, specified_classes=None):
        self.root = Path(root)
        if conversion:
            self.conversion_nii2npy(self.root)
        self.imgs, self.labels = self.get_img_list(self.root, valid_rate)

        if specified_classes is None:
            self.specified_classes = [0, 1, 2]
        else:
            assert len(self.get_classes_name(spec=False)) == len(specified_classes)
            self.specified_classes = specified_classes

        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()

        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self._split()

        self.img_channels = self.__getitem__(0)[0].shape[0]
        self.vis_idx = [300, 264, 179, 188, 42, 333, 48, 40, 147, 45,
                        17, 19, 24, 40, 119, 41, 60, 37, 68, 42,
                        29, 25, 253, 43, 38, 63, 176, 366, 27, 76,
                        13, 41, 57, 172, 45, 40, 55, 32, 21, 37,
                        67, 20, 164, 39, 37, 44, 70, 78, 30, 350,
                        41, 31, 166, 135, 32, 27, 42, 58, 29, 192,
                        18, 14, 34, 194, 27, 67, 190, 164, 274, 23]

    def _split(self):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        self.test_dataset = self

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
        cases.pop(160)
        self.split_case = int(np.round(len(cases) * valid_rate))
        for i in range(len(cases)):
            case = cases[i]
            imaging_dir = case / 'imaging'
            segmentation_dir = case / 'segmentation'
            assert imaging_dir.exists() and segmentation_dir.exists()

            imgs += sorted(list(imaging_dir.glob('*.npy')))
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

        idx = list(range(len(self.get_classes_name(spec=False))))
        masks = [np.where(label == i) for i in idx]

        spec_class_idx = []
        for i in self.specified_classes:
            if i not in spec_class_idx:
                spec_class_idx.append(i)

        for mask, spec_class in zip(masks, self.specified_classes):
            label[mask] = spec_class_idx.index(spec_class)

        spec_class_idx = []
        for i in self.specified_classes:
            if i not in spec_class_idx:
                spec_class_idx.append(i)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return {'image': image, 'label': label}

    def get_colormap(self, spec=True):
        cmap = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
        cmap = np.array(cmap, dtype=np.int)

        if not spec:
            return cmap
        else:
            spec_cmap = []
            for i in cmap[self.specified_classes]:
                if len(spec_cmap) == 0:
                    spec_cmap.append(i)
                else:
                    duplicate = False
                    for j in spec_cmap:
                        duplicate = duplicate or (i == j).all()
                    if not duplicate:
                        spec_cmap.append(i)
            return np.array(spec_cmap)

    def get_classes_name(self, spec=True):
        classes_name = np.array(['background', 'kidney', 'tumor'])

        if not spec:
            return classes_name
        else:
            spec_classes_name = []
            for i in classes_name[self.specified_classes]:
                if i not in spec_classes_name:
                    spec_classes_name.append(i)
            return spec_classes_name

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

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = np.load(str(img_path))

        label_path = self.labels[idx]
        label = np.load(str(label_path))

        data = {'image': img, 'label': label}
        if idx in self.train_indices and self.train_transform is not None:
            data = self.train_transform(data)
        elif idx in self.valid_indices and self.valid_transform is not None:
            data = self.valid_transform(data)

        data = self.default_transform(data)

        img = data['image']
        label = data['label']

        return img, label, idx

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    from utils.vis import imshow

    root = os.path.expanduser('~/dataset/kits19/data')
    dataset_ = kits19(root=root, valid_rate=0.2,
                      train_transform=None,
                      valid_transform=None, specified_classes=[0, 0, 2])

    train_loader, valid_loader, _ = dataset_.get_dataloader(batch_size=10)
    for batch_idx, (img, label, idx) in enumerate(train_loader):
        imgs, labels, _ = dataset_.vis_transform(imgs=img, labels=label, preds=None)
        imshow(title='kits19', imgs=(imgs[0], labels[0]))
