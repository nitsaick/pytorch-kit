import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SingleConv(in_ch, out_ch),
            SingleConv(out_ch, out_ch),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            SingleConv(out_ch, out_ch)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, out_ch, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = out_ch
        self.conv = SingleConv(out_ch, out_ch)
    
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCU(nn.Module):
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCU, self).__init__()
        self.RCU = nn.Sequential(
            RecurrentBlock(out_ch, t),
            RecurrentBlock(out_ch, t)
        )
        self.inc = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.inc(x)
        x1 = self.RCU(x)
        return x + x1


class AttentionBlock(nn.Module):
    def __init__(self, Fg, Fl, Fint):
        super(AttentionBlock, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(Fint)
        )
        
        self.Wx = nn.Sequential(
            nn.Conv2d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(Fint)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(Fint, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        sigma1 = g1 + x1
        sigma2 = self.relu(sigma1)
        alpha = self.psi(sigma2)
        
        return x * alpha


class PlainNode2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(PlainNode2, self).__init__()
        self.conv = SingleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x1, x), dim=1)
        x = self.conv(x)
        return x
