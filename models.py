# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.autograd as autograd
import torch.nn.functional as F

class ConvBnRelu3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, chls_group=12, is_bn=True, is_relu=True):
        super(ConvBnRelu3d, self).__init__()
        self.conv = nn.Conv3d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm3d(out_chl)
            # self.bn = nn.InstanceNorm3d(out_chl)
            # self.bn = nn.GroupNorm(num_groups=out_chl//chls_group, num_channels=out_chl)
        if is_relu is True:
            self.relu = nn.LeakyReLU(inplace=True)
            # self.relu = nn.GELU()
            # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class StackEncoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackEncoder, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool3d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out

class StackDecoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDecoder, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu3d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        )

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out  = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv_out = self.conv(torch.cat([up_out, conv_res], 1))
        return conv_out

class StackResEncoder3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResEncoder3d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.convx = None

        if in_chl != out_chl:

            self.convx = ConvBnRelu3d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x)+x)
        else:
            conv_out = F.leaky_relu(self.encode(x)+self.convx(x))
        
        down_out = F.max_pool3d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out

class StackResCenter3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResCenter3d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.convx = None

        if in_chl != out_chl:

            self.convx = ConvBnRelu3d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x)+x)
        else:
            conv_out = F.leaky_relu(self.encode(x)+self.convx(x))

        return conv_out

class StackResDecoder3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResDecoder3d, self).__init__()

        self.conv1 = ConvBnRelu3d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False)
        self.convx = ConvBnRelu3d(in_chl + out_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, up_in, conv_res):
        _, _, D, H, W = conv_res.size()
        up_out  = F.interpolate(up_in, size=(D, H, W), mode='trilinear', align_corners=True)
        conv1 = self.conv1(torch.cat([up_out, conv_res], 1))
        conv2 = self.conv2(conv1)
        convx = F.leaky_relu(conv2 + self.convx(torch.cat([up_out, conv_res], 1)))
        return convx

class UNet3d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(UNet3d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu3d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu3d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1),
                                    ConvBnRelu3d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu3d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

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
        res_out = F.leaky_relu(x+conv6)
        return res_out

class ResUNet3d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(ResUNet3d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu3d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackResEncoder3d(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackResEncoder3d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackResEncoder3d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackResEncoder3d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = StackResCenter3d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)  # 32

        self.up4 = StackResDecoder3d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder3d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder3d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder3d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu3d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

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
        res_out = F.leaky_relu(x+conv6)
        return res_out

class StackDenseEncoder3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseEncoder3d, self).__init__()

        self.conv1 = ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu3d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu3d(in_chl+out_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:

            self.convx = ConvBnRelu3d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.leaky_relu(conv3+x)
        else:
            convx = F.leaky_relu(conv3+self.convx(x))

        down_out = F.max_pool3d(convx, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return convx, down_out

class StackDenseBlock3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseBlock3d, self).__init__()

        self.conv1 = ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu3d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu3d(in_chl+out_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:

            self.convx = ConvBnRelu3d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.leaky_relu(conv3+x)
        else:
            convx = F.leaky_relu(conv3+self.convx(x))

        return convx

class DenseUNet3d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(DenseUNet3d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu3d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackDenseEncoder3d(self.model_chl, self.model_chl, kernel_size=3)
        self.down2 = StackDenseEncoder3d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)
        self.down3 = StackDenseEncoder3d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)
        self.down4 = StackDenseEncoder3d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)

        self.center = StackDenseBlock3d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)

        self.up4 = StackResDecoder3d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder3d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder3d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder3d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu3d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

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
        res_out = F.leaky_relu(x+conv6)
        return res_out