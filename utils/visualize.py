import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.evaluation import post_proc


def batch_numpy_to_plt(img):
    return img.transpose((0, 2, 3, 1))

def numpy_to_plt(img):
    return img.transpose((1, 2, 0))

def create_fig(shape):
    fig, _ = plt.subplots(shape[0], shape[1], figsize=(shape[1] * 3, shape[0] * 3), dpi=150, sharex=True, sharey=True)
    return fig

def imshow(imgs, fig, main_title=None, shape=None, sub_title=None, cmap=None, transpose=False):
    if fig is None:
        fig = create_fig(shape)

    if type(imgs) is tuple:
        num = len(imgs)
        assert shape[0] * shape[1] == num
        if type(sub_title) is not tuple:
            sub_title = (sub_title,) * num
        else:
            assert len(sub_title) == num

        if type(cmap) is not tuple:
            cmap = (cmap,) * num
        else:
            assert len(cmap) == num

        fig.suptitle(main_title)

        axes = fig.get_axes()
        for i in range(shape[0]):
            for j in range(shape[1]):
                index = i * shape[1] + j
                axes[index].set_title(sub_title[index])
                img = numpy_to_plt(imgs[index]) if transpose is True else imgs[index]
                axes[index].imshow(img, cmap[index])

    else:
        if transpose: imgs = numpy_to_plt(imgs)
        plt.figure(num=main_title)
        plt.suptitle(main_title)
        plt.title(sub_title)
        plt.imshow(imgs, cmap)

    plt.ion()
    plt.show()
    plt.pause(1)

def showone(img, window_name=0):

    plt.ion()
    plt.figure(num=window_name)
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.pause(1)


def show_two_img(img1, img2, window_name=0):
    plt.ion()
    plt.figure(num=window_name)

    plt.subplot(1, 2, 1)
    plt.title('Predict')
    plt.imshow(img1, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Ground Trust')
    plt.imshow(img2, cmap=None)

    plt.show()
    plt.pause(1)


def show_full(img, label, pred, threshold, title=None):
    if type(img).__module__ != np.__name__:
        img = img.cpu().numpy()
    if type(label).__module__ != np.__name__:
        label = label.cpu().numpy()
    if type(pred).__module__ != np.__name__:
        pred = pred.cpu().detach().numpy()

    thres = post_proc(pred, threshold)

    plt.ion()

    if title is not None:
        plt.figure(num=title)
    else:
        plt.figure(num=0)

    if title is not None:
        plt.suptitle(title)

    plt.subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Ground Truth')
    plt.imshow(label, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('Predict')
    plt.imshow(pred, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Threshold {}'.format(threshold))
    plt.imshow(thres, cmap='gray')

    plt.show()
    plt.pause(1)


if __name__ == '__main__':
    img = torch.rand(50, 50)
    device = torch.device('cuda:0')
    img = img.to(device)
    fig = create_fig((2, 2))
    imshow((img, img, img, img), fig, 'Test', (2, 2), ('a', 'b', 'b', 'b'))
