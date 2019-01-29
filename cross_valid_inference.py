from optparse import OptionParser

import network
from dataset.segmentation import *
from dataset.transform import random_flip_transform
from utils.metrics import Evaluator
from utils.switch import switch
from utils.vis import imshow
from utils.crf import dense_crf_batch


def get_args():
    parser = OptionParser()
    parser.add_option('--batch-size', dest='batch_size', default=1, type='int', help='batch size')
    parser.add_option('--gpu', dest='gpu', default=0, type='int', help='gpu of index, -1 is cpu')
    parser.add_option('--log-dir', dest='log_dir', default='~/runs/', help='tensorboard log dir')
    parser.add_option('--dataset-root', dest='dataset_root', default='~/dataset/', help='dataset root')
    parser.add_option('--dataset-name', dest='dataset_name', default='xVertSeg',
                      help='dataset name: xVertSeg')
    parser.add_option('--network', dest='network_name', default='UNet',
                      help='network name: UNet, R2UNet, AttUNet, AttR2UNet, IDANet, TUNet')
    parser.add_option('--base-ch', dest='base_ch', default=64, type='int', help='base channels')
    parser.add_option('--cross-valid', dest='cross_valid_k', default=1, type='int', help='cross valid k')
    parser.add_option('--split', dest='split', default=1, type='int', help='split')
    
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    
    device = 'cpu'
    if args.gpu != -1:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    dataset_root = os.path.expanduser(os.path.join(args.dataset_root, args.dataset_name))
    
    for case in switch(args.dataset_name):
        if case('xVertSeg'):
            dataset = xVertSegFold(root=dataset_root, cross_valid_k=args.cross_valid_k * args.split,
                                   transform=random_flip_transform, resume=False, log_dir=args.log_dir, step=0)
            break
        
        if case():
            raise AssertionError('Unknown dataset name.')
    
    evaluator = Evaluator(dataset.num_classes)
    for step in range(args.cross_valid_k * args.split):
        
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
        cp_file = os.path.join(args.log_dir, str(step // args.split + 1) + '/checkpoint/best.pth')
        checkpoint = torch.load(cp_file, map_location=device)
        net.load_state_dict(checkpoint['net'])
        
        net.eval()
        torch.set_grad_enabled(False)
        
        dataset.cross_valid_step(step)
        _, valid_loader, _ = dataset.get_dataloader(args.batch_size)

        acc_list = []
        crf_acc_list = []
        
        for batch_idx, (imgs, labels) in enumerate(valid_loader):
            imgs = imgs.to(device)
            outputs = net(imgs)
            
            np_outputs = outputs.cpu().detach().numpy().argmax(axis=1)
            np_labels = labels.cpu().detach().numpy()
            evaluator.reset()
            evaluator.add_batch(np_outputs, np_labels)
            acc = evaluator.eval('dc')
            acc_list.append(acc)

            imgs = imgs.cpu().detach().numpy() * 255
            imgs_color = np.concatenate((imgs, imgs, imgs), axis=1).astype(np.uint8)
            
            # np_outputs = np.expand_dims(np_outputs, axis=1)
            # bg = 1 - np_outputs
            # np_outputs = np.concatenate((bg, np_outputs), axis=1)
            # outputs_crf = dense_crf_batch(imgs_color, np_outputs)[:, 1]
            #
            # evaluator.reset()
            # evaluator.add_batch(outputs_crf, np_labels)
            # crf_acc = evaluator.eval('dc')
            # crf_acc_list.append(crf_acc)
            
            
            if batch_idx == 0:
                vis_imgs, vis_labels, vis_outputs = dataset.vis_transform(imgs, labels, outputs)
                imshow(title='Valid', imgs=(vis_imgs[0], vis_labels[0], vis_outputs[0]), shape=(1, 3),
                       subtitle=('image', 'label', 'predict'))
        
        avg = np.average(acc_list)
        std = np.std(acc_list)
        print('Step {} avg: {}, std: {}'.format(step + 1, avg, std))

        # crf_avg = np.average(crf_acc_list)
        # crf_std = np.std(crf_acc_list)
        # print('Step {} avg: {}, std: {}'.format(step + 1, crf_avg, crf_std))