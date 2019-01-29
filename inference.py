from optparse import OptionParser

import network
from dataset.segmentation import *
from dataset.transform import random_flip_transform
from utils.metrics import Evaluator
from utils.switch import switch
from utils.vis import imshow


def get_args():
    parser = OptionParser()
    parser.add_option('--batch-size', dest='batch_size', default=1, type='int', help='batch size')
    parser.add_option('--gpu', dest='gpu', default=0, type='int', help='gpu of index, -1 is cpu')
    parser.add_option('--cp', dest='cp_file', default='', help='checkpoint file path')
    parser.add_option('--log-dir', dest='log_dir', default='~/runs/', help='tensorboard log dir')
    parser.add_option('--dataset-root', dest='dataset_root', default='~/dataset/', help='dataset root')
    parser.add_option('--dataset-name', dest='dataset_name', default='SpineSeg',
                      help='dataset name: SpineSeg, xVertSeg, VOC2012Seg')
    parser.add_option('--network', dest='network_name', default='UNet',
                      help='network name: UNet, R2UNet, AttUNet, AttR2UNet, IDANet, TUNet')
    parser.add_option('--base-ch', dest='base_ch', default=64, type='int', help='base channels')
    
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    
    device = 'cpu'
    if args.gpu != -1:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    dataset_root = os.path.expanduser(os.path.join(args.dataset_root, args.dataset_name))
    
    for case in switch(args.dataset_name):
        if case('SpineSeg'):
            dataset = SpineSeg(root=dataset_root, valid_rate=0.2, transform=random_flip_transform,
                               shuffle=False, resume=True, log_dir=args.log_dir)
            break
        
        if case('xVertSeg'):
            dataset = xVertSeg(root=dataset_root, valid_rate=0.25, transform=random_flip_transform,
                               shuffle=False, resume=True, log_dir=args.log_dir)
            break
        
        if case('VOC2012Seg'):
            dataset = VOC2012Seg(root=dataset_root, valid_rate=0.5, transform=None, transform_params=None,
                                 shuffle=False, resume=True, log_dir=args.log_dir, resize=(256, 256))
            break
        
        if case():
            raise AssertionError('Unknown dataset name.')
    
    for case in switch(args.network_name):
        if case('UNet'):
            net = network.UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('R2UNet'):
            net = network.R2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('AttUNet'):
            net = network.AttUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('AttR2UNet'):
            net = network.AttR2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('IDANet'):
            net = network.IDANet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=args.base_ch)
            break
        
        if case('TUNet'):
            net = network.Tunable_UNet(in_channels=1, n_classes=1, depth=5, wf=6,
                                       padding=True, batch_norm=True, up_mode='upconv')
            break
        
        if case():
            raise AssertionError('Unknown network name.')
    
    net.to(device)
    checkpoint = torch.load(args.cp_file, map_location=device)
    net.load_state_dict(checkpoint['net'])
    
    net.eval()
    torch.set_grad_enabled(False)
    
    _, valid_loader, _ = dataset.get_dataloader(args.batch_size)
    evaluator = Evaluator(dataset.num_classes)
    
    acc = 0.0
    for batch_idx, (imgs, labels) in enumerate(valid_loader):
        imgs = imgs.to(device)
        outputs = net(imgs)
        
        np_outputs = outputs.cpu().detach().numpy().argmax(axis=1)
        np_labels = labels.cpu().detach().numpy()
        evaluator.add_batch(np_outputs, np_labels)
        
        if batch_idx == 0:
            vis_imgs, vis_labels, vis_outputs = dataset.vis_transform(imgs, labels, outputs)
            imshow(title='Valid', imgs=(vis_imgs[0], vis_labels[0], vis_outputs[0]), shape=(1, 3),
                   subtitle=('image', 'label', 'predict'))
    
    valid_acc = evaluator.eval('dc')
    print('Valid Acc: {}'.format(valid_acc))
