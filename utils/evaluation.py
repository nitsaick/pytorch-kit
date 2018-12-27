import numpy as np
import torch


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def post_proc(img, threshold):
    if type(img).__module__ != np.__name__:
        img = img.cpu().detach().numpy()

    img = normalize(img)
    return img > threshold


def dice_coef(imgs, labels, thres):
    if type(thres) is tuple:
        assert len(thres) == 2
        imgs_thres = thres[0]
        labels_thres = thres[1]
    else:
        imgs_thres = labels_thres = thres
    imgs = post_proc(imgs, imgs_thres)
    labels = post_proc(labels, labels_thres)

    dice = np.sum(imgs[labels]) * 2.0 / (np.sum(imgs) + np.sum(labels))
    return dice


if __name__ == '__main__':
    a = torch.rand(4, 1, 3, 1)
    b = torch.rand(4, 1, 3, 1)
    thres = 0.5

    dice_coef(a, b, thres)
