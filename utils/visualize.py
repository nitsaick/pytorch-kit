import matplotlib.pyplot as plt
import torch


def numpy_to_plt(img):
    return img.transpose((1, 2, 0))


def imshow(main_title, imgs, shape=None, sub_title=None, cmap=None, transpose=False):
    if type(imgs) is tuple:
        num = len(imgs)
        if shape is not None:
            assert shape[0] * shape[1] == num
        else:
            shape = (1, num)

        if type(sub_title) is not tuple:
            sub_title = (sub_title,) * num
        else:
            assert len(sub_title) == num
        
        if type(cmap) is not tuple:
            cmap = (cmap,) * num
        else:
            assert len(cmap) == num
            
        fig = plt.figure(num=main_title, figsize=(shape[1] * 3, shape[0] * 3))
        fig.clf()
        fig.suptitle(main_title)

        fig.subplots(shape[0], shape[1], sharex=True, sharey=True)
        axes = fig.get_axes()

        for i in range(shape[0]):
            for j in range(shape[1]):
                idx = i * shape[1] + j
                axes[idx].set_title(sub_title[idx])
                
                cm = cmap[idx]
                img = imgs[idx]
                if cmap[idx] is None and len(img.shape) == 3 and img.shape[0] == 1:
                    cm = 'gray'
                    img = img.reshape((img.shape[1], img.shape[2]))
                
                img = numpy_to_plt(img) if transpose is True else img
                axes[idx].imshow(img, cm)

    else:
        if transpose: imgs = numpy_to_plt(imgs)
        plt.figure(num=main_title)
        plt.suptitle(main_title)
        plt.title(sub_title)
        plt.imshow(imgs, cmap)

    plt.ion()
    plt.show()
    plt.pause(1)


if __name__ == '__main__':
    img = torch.rand(50, 50)
    imshow('Test', (img, img, img, img), (2, 2), ('a', 'b', 'c', 'd'))
