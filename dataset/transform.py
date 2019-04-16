from random import random

import cv2
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Compose,
    GridDistortion,
    OpticalDistortion,
    RandomCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    RandomScale
)

__all__ = ['to_tensor', 'random_flip_transform', 'random_crop_transform', 'random_scale_crop',
           'medical_transform', 'real_world_transform']


def to_tensor(data):
    image, label = data['image'], data['label']
    image = F.to_tensor(image)
    label = F.to_tensor(label)
    return {'image': image, 'label': label}


def to_numpy(data):
    image, label = data['image'], data['label']
    image = np.array(image)
    label = np.array(label)
    label = label.reshape((*label.shape, 1))
    return {'image': image, 'label': label}


def random_flip_transform(data):
    image, label = data['image'], data['label']

    # Random horizontal flipping
    if random() > 0.5:
        image = F.hflip(image)
        label = F.hflip(label)

    # Random vertical flipping
    if random() > 0.5:
        image = F.vflip(image)
        label = F.vflip(label)

    if random() > 0.5:
        gamma = random() * 1 + 0.5
        image = F.adjust_gamma(image, gamma)

    if random() > 0.5:
        contrast_factor = random() * 1 + 0.5
        image = F.adjust_contrast(image, contrast_factor)

    if random() > 0.5:
        angle = random() * 20 - 10
        translate = (0, 0)
        scale = random() * 0.2 + 0.9
        shear = 0
        image = F.affine(image, angle, translate, scale, shear)
        label = F.affine(label, angle, translate, scale, shear)

    data = {'image': image, 'label': label}

    return data


def random_crop_transform(img, label, transform_params):
    width, height = img.size
    padh = width - height if width > height else 0
    padw = height - width if height > width else 0
    img = ImageOps.expand(img, border=(padw // 2, padh // 2, padw // 2, padh // 2), fill=0)
    label = ImageOps.expand(label, border=(padw // 2, padh // 2, padw // 2, padh // 2), fill=0)

    oh, ow = transform_params
    img = img.resize((ow, oh), Image.BILINEAR)
    label = label.resize((ow, oh), Image.NEAREST)

    img, label = random_flip_transform(img, label, transform_params)
    return img, label


class random_scale_crop:
    def __init__(self, output_size, scale_range=0.1, type='train'):
        if isinstance(output_size, (tuple, list)):
            self.output_size = output_size  # (h, w)
        else:
            self.output_size = (output_size, output_size)

        self.scale_range = scale_range
        self.type = type

    def __call__(self, data):
        img, label = data['image'], data['label']

        img = np.array(img)
        label = np.array(label)

        img_size = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
        crop_size = self.output_size[0] if self.output_size[0] < self.output_size[1] else self.output_size[1]
        scale = crop_size / img_size - 1
        if scale < 0:
            scale_limit = (scale - self.scale_range, scale + self.scale_range)
        else:
            scale_limit = (-self.scale_range, scale + self.scale_range)

        if self.type == 'train':
            aug = Compose([
                RandomScale(scale_limit=scale_limit, p=1),
                PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                OneOf([
                    RandomCrop(height=self.output_size[0], width=self.output_size[1], p=1),
                    CenterCrop(height=self.output_size[0], width=self.output_size[1], p=1)
                ], p=1),
            ])
        elif self.type == 'valid':
            aug = Compose([
                PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                CenterCrop(height=self.output_size[0], width=self.output_size[1], p=1)
            ])

        data = aug(image=img, mask=label)
        img, label = data['image'], data['mask']

        if len(img.shape) == 2:
            img = img.reshape((*img.shape, 1))
        if len(label.shape) == 2:
            label = label.reshape((*label.shape, 1))

        data = {'image': img, 'label': label}
        return data


class medical_transform:
    def __init__(self, output_size, scale_range, type):
        if isinstance(output_size, (tuple, list)):
            self.size = output_size  # (h, w)
        else:
            self.size = (output_size, output_size)

        self.scale_range = scale_range
        self.type = type

    def __call__(self, data):
        aug = random_scale_crop(output_size=self.size, scale_range=self.scale_range, type=self.type)
        data = aug(data)

        img, label = data['image'], data['label']

        img = np.array(img)
        label = np.array(label)

        if self.type == 'train':
            aug = Compose([
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                OneOf([
                    GridDistortion(p=1),
                    OpticalDistortion(p=1, distort_limit=1, shift_limit=10)
                ], p=0.5),
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.5)
            ])
            data = aug(image=img, mask=label)
            img, label = data['image'], data['mask']

        if len(img.shape) == 2:
            img = img.reshape((*img.shape, 1))
        if len(label.shape) == 2:
            label = label.reshape((*label.shape, 1))

        data = {'image': img, 'label': label}
        return data


class real_world_transform:
    def __init__(self, output_size, scale_range, type):
        if isinstance(output_size, (tuple, list)):
            self.size = output_size  # (h, w)
        else:
            self.size = (output_size, output_size)

        self.scale_range = scale_range
        self.type = type

    def __call__(self, data):
        aug = random_scale_crop(output_size=self.size, scale_range=self.scale_range, type=self.type)
        data = aug(data)

        img, label = data['image'], data['label']

        img = np.array(img)
        label = np.array(label)

        if self.type == 'train':
            aug = Compose([
                HorizontalFlip(p=0.25),
                CLAHE(p=0.25),
                RandomBrightnessContrast(p=0.25),
                RandomGamma(p=0.25)
            ])
            data = aug(image=img, mask=label)
            img, label = data['image'], data['mask']

        if len(img.shape) == 2:
            img = img.reshape((*img.shape, 1))
        if len(label.shape) == 2:
            label = label.reshape((*label.shape, 1))

        data = {'image': img, 'label': label}
        return data
