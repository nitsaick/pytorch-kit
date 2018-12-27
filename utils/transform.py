from random import random
import torchvision.transforms.functional as F

def random_flip_transform(image, label):
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
        image = F.adjust_gamma(image, contrast_factor)

    if random() > 0.5:
        angle = random() * 20 - 10
        translate = (0, 0)
        scale = random() * 0.2 + 0.9
        shear = 0
        image = F.affine(image, angle, translate, scale, shear)
        label = F.affine(label, angle, translate, scale, shear)

    # Transform to tensor
    image = F.to_tensor(image)
    label = F.to_tensor(label)
    return image, label