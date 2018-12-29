from random import random

import torchvision.transforms.functional as F


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
        img = F.adjust_gamma(img, contrast_factor)
    
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
    i, j = height, width
    h, w = transform_params
    
    while i + h >= height or j + w >= width:
        i = random() * height
        j = random() * width
    
    img = F.crop(img, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    
    return img, label
