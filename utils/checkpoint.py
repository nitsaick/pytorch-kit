import glob
import os

import torch


def save(epoch, net, optimizer, root):
    torch.save({'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, root)


def load_latest(root):
    files = glob.glob(root + '*.pth')
    if len(files):
        latest_file = max(files, key=os.path.getctime)
        latest_checkpoint = torch.load(latest_file)
        return latest_checkpoint
    else:
        raise FileNotFoundError('No checkpoint file in "{}"'.format(root))


def load_file(root):
    checkpoint = torch.load(root)
    return checkpoint


def load_params(root, net, optimizer, device):
    checkpoint = load_file(root)
    epoch = checkpoint['epoch'] + 1
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return net, optimizer, epoch


if __name__ == '__main__':
    cp = load_file('./checkpoint/')
