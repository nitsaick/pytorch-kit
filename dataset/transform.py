from random import random
from PIL import Image, ImageOps

import torchvision.transforms.functional as F
import numpy as np
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
)
from albumentations.pytorch import ToTensor


def to_tensor(img, label):
    img = F.to_tensor(img)
    label = F.to_tensor(label)
    return img, label


def random_flip_transform(img, label, transform_params):
    # Random horizontal flipping
    if random() > 0.5:
        img = F.hflip(img)
        label = F.hflip(label)

    # Random vertical flipping
    if random() > 0.5:
        img = F.vflip(img)
        label = F.vflip(label)

    if random() > 0.5:
        gamma = random() * 1 + 0.5
        img = F.adjust_gamma(img, gamma)

    if random() > 0.5:
        contrast_factor = random() * 1 + 0.5
        img = F.adjust_contrast(img, contrast_factor)

    if random() > 0.5:
        angle = random() * 20 - 10
        translate = (0, 0)
        scale = random() * 0.2 + 0.9
        shear = 0
        img = F.affine(img, angle, translate, scale, shear)
        label = F.affine(label, angle, translate, scale, shear)

    return img, label


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


def mix_transform(img, label, _):
    img = np.array(img)
    label = np.array(label)

    height, width = img.shape[:2]
    min_v = min(height, width)

    aug = Compose([
        OneOf([
            RandomSizedCrop(min_max_height=(min_v // 2, min_v), height=height, width=width,
                            w2h_ratio=width / height, p=0.5),
            PadIfNeeded(min_height=height, min_width=width, p=0.5)
        ], p=1),
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

    img = img.reshape((*img.shape, 1))
    label = label.reshape((*label.shape, 1))

    return img, label

def mix_transform2(img, label, transform_params):
    crop_size = 256
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    img = img.resize((ow, oh), Image.BILINEAR)
    label = label.resize((ow, oh), Image.NEAREST)

    img = np.array(img)
    label = np.array(label)

    if transform_params == 'train':
        aug = Compose([
            OneOf([
                RandomSizedCrop(min_max_height=(crop_size // 1.2, crop_size), height=crop_size, width=crop_size,
                                w2h_ratio=1, p=1),
                CenterCrop(p=1, height=crop_size, width=crop_size)
            ], p=1),
            # VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            CLAHE(p=0.5),
            RandomBrightnessContrast(p=0.5),
            RandomGamma(p=0.5)
        ])
    elif transform_params == 'valid':
        aug = CenterCrop(p=1, height=crop_size, width=crop_size)

    data = aug(image=img, mask=label)
    img, label = data['image'], data['mask']

    # img = img.reshape((*img.shape, 1))
    label = label.reshape((*label.shape, 1))

    return img, label
