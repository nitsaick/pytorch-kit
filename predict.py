from optparse import OptionParser
import csv

import torch
import torchvision

import utils.checkpoint as cp
from dataset.SpineSeg import SpineSeg
from utils.evaluation import dice_coef
from network.unet import UNet


def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-d', '--dataset_root', dest='root',
                      default='./data/', help='dataset root')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    checkpoint_root = './checkpoint/'
    checkpoint = cp.load_file(checkpoint_root + 'best.pth')

    net = UNet(1, 1)
    # net = R2UNet()
    net.load_state_dict(checkpoint['net'])

    device = 'cpu'
    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = SpineSeg(args.root, resume=True, valid_rate=0.2)

    batch_size = 1
    _, valid_loader, test_loader = dataset.get_dataloader(batch_size)

    net = net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    total_acc = 0.0
    batch_num = 0
    thres = 0.5

    acc_log_filename = './acc.csv'
    csv_out_file = open(acc_log_filename, 'w', newline='')
    csv_out = csv.writer(csv_out_file)
    out_row = ('img', 'acc')
    csv_out.writerow(out_row)

    for batch_index, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        output = net(imgs)
        output = torch.sigmoid(output)

        acc = dice_coef(output, labels, thres)
        total_acc += acc
        batch_num += 1

        for i in range(len(output)):
            imglist = [imgs[i].cpu(), labels[i].cpu(), output[i].cpu()]
            result = torchvision.utils.make_grid(imglist, padding=10, normalize=True)

            filename = './pred/' + dataset.imgs[batch_index * batch_size + i].split('/')[-1]
            torchvision.utils.save_image(result, filename)

            out_row = (dataset.imgs[batch_index * batch_size + i].split('/')[-1], acc)
            csv_out.writerow(out_row)

    print('Avg acc: {:.5f}  \n'.format(total_acc / batch_num))
    csv_out_file.close()