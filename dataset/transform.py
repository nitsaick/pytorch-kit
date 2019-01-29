from random import random
from PIL import Image, ImageOps

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
