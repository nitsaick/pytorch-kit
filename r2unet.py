import torch
import torch.nn as nn


# Recurrent Residual Convolutional Units
class rrcu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(rrcu, self).__init__()
        self.conv_forward1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv_forward2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.conv_recurrent1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.conv_recurrent2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.seq1 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        xf1 = self.conv_forward1(x)
        xr1 = self.conv_forward1(x) + self.conv_recurrent1(xf1)
        xr2 = self.conv_forward1(x) + self.conv_recurrent1(xr1)
        xr3 = self.conv_forward1(x) + self.conv_recurrent1(xr2)

        xf2 = self.conv_forward2(xr3)
        xr4 = self.conv_forward2(xr3) + self.conv_recurrent2(xf2)
        xr5 = self.conv_forward2(xr3) + self.conv_recurrent2(xr4)
        xr6 = self.conv_forward2(xr3) + self.conv_recurrent2(xr5)

        xr = self.seq1(xr6)

        out = xf1 + xr

        out = self.seq2(out)

        return out


class in_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(in_conv, self).__init__()
        self.conv = rrcu(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            rrcu(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = rrcu(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class R2UNet(nn.Module):
    def __init__(self, channels_num=1, classes_num=1):
        super(R2UNet, self).__init__()
        self.inc = in_conv(channels_num, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = out_conv(64, classes_num)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
