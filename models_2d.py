# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F

# input (Tensor)
# pad (tuple)
# mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
# value – fill value for 'constant' padding. Default: 0

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None
    
        if is_bn is True:
            self.bn = nn.BatchNorm2d(out_chl, eps=1e-4)
        if is_relu is True:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class StackEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackEncoder2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out

class StackDecoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDecoder2d, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu2d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        )

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out  = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv_out = self.conv(torch.cat([up_out, conv_res], 1))
        return conv_out

class StackDenseEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseEncoder2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu2d(in_chl+out_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:

            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.relu(conv3+x)
        else: 
            convx = F.relu(conv3+self.convx(x))

        down_out = F.max_pool2d(convx, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return convx, down_out

class StackDenseBlock2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseBlock2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu2d(in_chl+out_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:

            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.relu(conv3+x)
        else: 
            convx = F.relu(conv3+self.convx(x))

        return convx

class StackResDecoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResDecoder2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False)
        self.convx = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out  = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv1 = self.conv1(torch.cat([up_out, conv_res], 1))
        conv2 = self.conv2(conv1)
        convx = F.relu(conv2 + self.convx(torch.cat([up_out, conv_res], 1)))
        return convx

class UNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(UNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder2d(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1), 
                                    ConvBnRelu2d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu2d(self.model_chl, 1, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1) 
        conv3, d3 = self.down3(d2) 
        conv4, d4 = self.down4(d3) 
        conv5 = self.center(d4) 
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3) 
        up2 = self.up2(up3, conv2) 
        up1 = self.up1(up2, conv1) 
        conv6 = self.end(up1)
        res_out = F.relu(x+conv6)
        return res_out

class DenseUNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=128):
        super(DenseUNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackDenseEncoder2d(self.model_chl, self.model_chl, kernel_size=3)
        self.down2 = StackDenseEncoder2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)
        self.down3 = StackDenseEncoder2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)
        self.down4 = StackDenseEncoder2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)

        self.center = StackDenseBlock2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)

        self.up4 = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu2d(self.model_chl, 1, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1) 
        conv3, d3 = self.down3(d2) 
        conv4, d4 = self.down4(d3) 
        conv5 = self.center(d4) 
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3) 
        up2 = self.up2(up3, conv2) 
        up1 = self.up1(up2, conv1) 
        conv6 = self.end(up1)
        res_out = F.relu(x+conv6)
        return res_out