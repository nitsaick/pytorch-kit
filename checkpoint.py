import torch
import glob
import os


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


if __name__ == '__main__':
    cp = load_file('./checkpoint/')
    cp = load_file('./checkpoint/1.pth')
