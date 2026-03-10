#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>


from torch import nn
# from config import cfg
import torch.nn.functional as F
import torch 
import numpy as np
def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.ReLU(inplace=True)
    )

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
        nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer, 
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out

def init_upconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)

def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output

def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook

class STFAN_Net(nn.Module):
    def __init__(self,input_channel=3):
        super(STFAN_Net, self).__init__()
        #############################
        # Deblurring Branch
        #############################
        # encoder
        ks = 3
        ks_2d = 5
        ch1 = 64
        ch2 = 64
        ch3 = 128

        self.conv1_1 = conv(input_channel, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.conv1_3 = resnet_block(ch1, kernel_size=ks)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.conv2_3 = resnet_block(ch2, kernel_size=ks)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.conv3_3 = resnet_block(ch3, kernel_size=ks)

        # decoder
        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks)

        self.img_prd = conv(ch1, 3, kernel_size=ks)

        #############################
        # Kernel Prediction Branch
        #############################

        # kernel network
        self.kconv1_1 = conv(input_channel, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.kconv1_3 = resnet_block(ch1, kernel_size=ks)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.kconv2_3 = resnet_block(ch2, kernel_size=ks)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = resnet_block(ch3, kernel_size=ks)
        self.kconv3_3 = resnet_block(ch3, kernel_size=ks)

        self.fac_warp = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        self.kconv4 = conv(ch3 * ks_2d ** 2, ch3, kernel_size=1)

        self.fac_deblur = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

    def forward(self, img_blur):
        merge = img_blur

        #############################
        # Kernel Prediction Branch
        #############################
        # kernel network
        kconv1 = self.kconv1_3(self.kconv1_2(self.kconv1_1(merge)))
        kconv2 = self.kconv2_3(self.kconv2_2(self.kconv2_1(kconv1)))
        kconv3 = self.kconv3_3(self.kconv3_2(self.kconv3_1(kconv2)))
        # fac

        kernel_deblur = self.fac_deblur(torch.cat([kconv3],1))

        #############################
        # Deblurring Branch
        #############################
        # encoder blur
        conv1_d = self.conv1_1(img_blur)
        conv1_d = self.conv1_3(self.conv1_2(conv1_d))

        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_3(self.conv2_2(conv2_d))

        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_3(self.conv3_2(conv3_d))

        # conv3_d_k = self.kconv_deblur(conv3_d, kernel_deblur)


        conv3 = torch.cat([conv3_d],1)

        # decoder
        upconv2 = self.upconv2_1(self.upconv2_2(self.upconv2_u(conv3)))
        upconv1 = self.upconv1_1(self.upconv1_2(self.upconv1_u(upconv2)))
        output_img = self.img_prd(upconv1)
        # print('output_img=',output_img.shape)
        return output_img


# my_model = STFAN_Net(input_channel=3)
# _input = torch.zeros(2,3,16,16)
# _output = my_model(_input)
# print('_input=',_input.shape)
# print('_output=',_output.shape)