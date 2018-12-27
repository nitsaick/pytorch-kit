import numpy as np
import pydensecrf.densecrf as dcrf
import torch

import utils.checkpoint as cp
from dataset.xVertSeg import xVertSeg
from utils.evaluation import dice_coef
from network.network import U_Net

from utils.visualize import *


def dense_crf(img, output):
    c, h, w = output.shape

    unary = -np.log(output)
    unary = unary.reshape(c, -1)
    unary = np.ascontiguousarray(unary)

    img = img.transpose(1, 2, 0)
    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(unary)

    # d.addPairwiseEnergy()
    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    q = d.inference(5)
    q = np.array(q).reshape(c, h, w)

    return q

# img: b, c=3, h ,w
# output b, c=2, h ,w
def dense_crf_batch(img, output):
    assert img.shape[0] == output.shape[0]
    batch_size = img.shape[0]
    output_crf = np.zeros_like(output)

    for i in range(batch_size):
        output_crf[i] = dense_crf(img[i], output[i])

    return output_crf


if __name__ == '__main__':
    checkpoint_root = 'E:/Dropbox/Workspace/pytorch/runs/xVertSeg/checkpoint/'
    checkpoint = cp.load_file(checkpoint_root + 'best.pth')

    net = U_Net(1, 1)
    # net = R2UNet()
    net.load_state_dict(checkpoint['net'])

    device = 'cuda:0'

    dataset = xVertSeg('./data/xVertSeg/', resume=True, valid_rate=0.25,
                       log_dir='E:/Dropbox/Workspace/pytorch/runs/xVertSeg')

    batch_size = 1
    _, valid_loader, _ = dataset.get_dataloader(batch_size)

    net = net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    total_acc = 0.0
    total_acc_crf = 0.0
    batch_num = 0
    thres = 0.5

    fig = create_fig((2, 2))

    for batch_index, (imgs, labels) in enumerate(valid_loader):
        imgs = imgs.to(device)
        outputs = net(imgs)
        outputs = torch.sigmoid(outputs)

        acc = dice_coef(outputs, labels, thres)
        total_acc += acc

        imgs = imgs.cpu().detach().numpy() * 255
        imgs_color = np.concatenate((imgs, imgs, imgs), axis=1).astype(np.uint8)

        outputs = outputs.cpu().detach().numpy()
        bg = 1 - outputs
        outputs = np.concatenate((bg, outputs), axis=1)

        outputs_crf = dense_crf_batch(imgs_color, outputs)[:, 1]
        outputs_crf = np.expand_dims(outputs_crf, axis=1)

        acc_crf = dice_coef(outputs_crf, labels, thres)
        total_acc_crf += acc_crf

        if acc_crf - acc > 0.05 or acc_crf - acc < -0.05:
            imshow((imgs[0][0], labels[0][0], outputs[0][1], outputs_crf[0][0]), None, str(acc_crf - acc), (2, 2), ('Img', 'Label', 'Output', 'CRF'), 'gray')

        batch_num += 1
        print('before: {}'.format(acc))
        print('after : {}'.format(acc_crf))
        print('diff  : {}\n'.format(acc_crf - acc))

    print('Avg acc    : {:.5f}  \n'.format(total_acc / batch_num))
    print('Avg acc_crf: {:.5f}  \n'.format(total_acc_crf / batch_num))
