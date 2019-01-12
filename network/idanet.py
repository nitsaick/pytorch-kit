import torch
import torch.nn as nn

from .subnet import DoubleConv, PlainNode2


class IDANet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=64):
        super(IDANet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = DoubleConv(in_ch=in_ch, out_ch=base_ch * 2)
        self.Conv4 = DoubleConv(in_ch=base_ch * 2, out_ch=base_ch * 4)
        self.Conv8 = DoubleConv(in_ch=base_ch * 4, out_ch=base_ch * 8)
        self.Conv16 = DoubleConv(in_ch=base_ch * 8, out_ch=base_ch * 16)
        self.Conv32 = DoubleConv(in_ch=base_ch * 16, out_ch=base_ch * 32)

        self.agn2_1 = PlainNode2(in_ch=base_ch * (2 + 4), out_ch=base_ch * 2)
        self.agn2_2 = PlainNode2(in_ch=base_ch * (2 + 4), out_ch=base_ch * 2)
        self.agn2_3 = PlainNode2(in_ch=base_ch * (2 + 4), out_ch=base_ch * 2)
        self.agn2_4 = PlainNode2(in_ch=base_ch * (2 + 4), out_ch=base_ch * 2)

        self.agn4_1 = PlainNode2(in_ch=base_ch * (4 + 8), out_ch=base_ch * 4)
        self.agn4_2 = PlainNode2(in_ch=base_ch * (4 + 8), out_ch=base_ch * 4)
        self.agn4_3 = PlainNode2(in_ch=base_ch * (4 + 8), out_ch=base_ch * 4)

        self.agn8_1 = PlainNode2(in_ch=base_ch * (8 + 16), out_ch=base_ch * 8)
        self.agn8_2 = PlainNode2(in_ch=base_ch * (8 + 16), out_ch=base_ch * 8)

        self.agn16_1 = PlainNode2(in_ch=base_ch * (16 + 32), out_ch=base_ch * 16)

        self.outc = nn.Conv2d(base_ch * 2, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x2 = self.Conv2(x)

        x4 = self.maxpool(x2)
        x4 = self.Conv4(x4)

        x8 = self.maxpool(x4)
        x8 = self.Conv8(x8)

        x16 = self.maxpool(x8)
        x16 = self.Conv16(x16)

        x32 = self.maxpool(x16)
        x32 = self.Conv32(x32)

        x2 = self.agn2_1(x2, x4)
        x4 = self.agn4_1(x4, x8)
        x8 = self.agn8_1(x8, x16)
        x16 = self.agn16_1(x16, x32)

        x2 = self.agn2_2(x2, x4)
        x4 = self.agn4_2(x4, x8)
        x8 = self.agn8_2(x8, x16)

        x2 = self.agn2_3(x2, x4)
        x4 = self.agn4_3(x4, x8)

        x2 = self.agn2_4(x2, x4)

        out = self.outc(x2)

        return out
