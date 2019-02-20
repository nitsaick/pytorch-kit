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

__all__ = ['to_tensor', 'random_flip_transform', 'random_crop_transform', 'medical_transform', 'real_world_transform']


def to_tensor(data):
    image, label = data['image'], data['label']
    image = F.to_tensor(image)
    label = F.to_tensor(label)
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


class medical_transform:
    def __init__(self):
        pass

    def __call__(self, data):
        image, label = data['image'], data['label']
        
        image = np.array(image)
        label = np.array(label)
        
        height, width = image.shape[:2]
        min_v = min(height, width)
        
        aug = Compose([
            OneOf([
                RandomSizedCrop(min_max_height=(min_v // 1.2, min_v), height=height, width=width,
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
        
        data = aug(image=image, mask=label)
        image, label = data['image'], data['mask']
        
        image = image.reshape((*image.shape, 1))
        label = label.reshape((*label.shape, 1))
        
        data = {'image': image, 'label': label}
        
        return data


class real_world_transform:
    def __init__(self, output_size, type):
        self.size = output_size
        self.type = type
    
    def __call__(self, data):
        img, label = data['image'], data['label']
        
        crop_size = self.size
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
        
        if self.type == 'train':
            aug = Compose([
                OneOf([
                    RandomSizedCrop(min_max_height=(crop_size // 1.2, crop_size), height=crop_size, width=crop_size,
                                    w2h_ratio=1, p=1),
                    CenterCrop(p=1, height=crop_size, width=crop_size)
                ], p=1),
                HorizontalFlip(p=0.5),
                CLAHE(p=0.5),
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.5)
            ])
        elif self.type == 'valid':
            aug = CenterCrop(p=1, height=crop_size, width=crop_size)
        
        data = aug(image=img, mask=label)
        img, label = data['image'], data['mask']
        
        label = label.reshape((*label.shape, 1))
        
        data = {'image': img, 'label': label}
        
        return data
