# system modules
import os, argparse
from datetime import datetime

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import UCF101

# custom utilities
from dataloader import MNIST_Dataset, KTH_Dataset
from convlstmnet import ConvLSTMNet
from contextvpnet import ContextVPNet
import matplotlib.image as imgblin
import skimage

from torch.nn.init import kaiming_normal_, constant_

import numpy as np
from torch.utils.data import Dataset, DataLoader
from frozen_batchnorm import FrozenBatchNorm

# 3x3 convolution
def conv3x3(in_channels, out_channels):
    conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size = 3, padding = 1)
    torch.nn.init.xavier_normal_(conv.weight)
    return conv

class Block_Dec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block_Dec, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_channels * 2, out_channels)
        self.relu2 = nn.ReLU()


    def forward(self, x, res):
        out = self.conv1(res)
        out = self.relu1(out)
        out = torch.cat([x, out], dim=1)
        out = self.conv2(out)
        out = self.relu2(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')

        return out



class DecoderResLstm(nn.Module):
    def __init__(self, layers_input, lstm_layer):
        super(DecoderResLstm, self).__init__()

        self.layers_input = layers_input
        self.lstm_layer = lstm_layer
        self.layers = nn.ModuleDict()
        for l in range(len(layers_input)):
            lid = "l{}".format(l)
            self.layers[lid] = Block_Dec(self.layers_input[l], self.lstm_layer)

        self.layers["out"] = nn.Conv2d(self.lstm_layer, 1, kernel_size=1, padding=0, bias=False)
        torch.nn.init.xavier_normal_(self.layers["out"].weight)


    def forward(self, x, res):

        for l in range(len(self.layers_input)):
            lid = "l{}".format(l)
            x = self.layers[lid](x, res[l])

        x = self.layers["out"](x)

        return x


class Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50, self).__init__()

        res = torchvision.models.resnet50(pretrained=True, replace_stride_with_dilation = [True, True, True])

        #print(res)

        weight = res.conv1.weight.data
        weight1 = res.layer4[0].conv2.weight.data
        weight2 = res.layer4[0].downsample[0].weight.data

        res.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        res.conv1.weight[:, :3].data = weight

        # res.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        #
        # res.layer4[0].conv2.weight.data = weight1
        #
        # res.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, dilation=2, bias=False)
        #
        # res.layer4[0].downsample[0].weight.data = weight2


        # print(res)
        # print(res.layer4[0].conv2)

        # res.conv1.weight[:, :3] = weight
        # res.conv1.weight[:, 3:6] = weight
        #res.conv1.weight[:, 3:6] = res.conv1.weight[:, 0]

        self.net = nn.Sequential(*list(res.children())[:-2])

    def forward(self, x):
        res = []
        for layer in self.net:
            x = layer(x)
            res.append(x)
        #x = self.net(x)
            # print("x", x.size())

        return x, res


class Res34(nn.Module):
    def __init__(self, pretrained=True):
        super(Res34, self).__init__()

        res = torchvision.models.resnet34(pretrained=True, replace_stride_with_dilation=[False, False, False])

        #print(res)

        weight = res.conv1.weight.data

        weight1 = res.layer4[0].downsample[0].weight.data
        weight2 = res.layer3[0].downsample[0].weight.data
        weight3 = res.layer2[0].downsample[0].weight.data
        weight1_1 = res.layer4[0].conv1.weight.data
        weight2_1 = res.layer3[0].conv1.weight.data
        weight3_1 = res.layer2[0].conv1.weight.data

        res.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        res.conv1.weight[:, :3].data = weight

        # res.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        #
        # res.layer4[0].conv2.weight.data = weight1
        #
        res.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)

        res.layer3[0].downsample[0].weight.data = weight2

        res.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)

        res.layer4[0].downsample[0].weight.data = weight1

        res.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)

        res.layer2[0].downsample[0].weight.data = weight3

        res.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)

        res.layer3[0].conv1.weight.data = weight2_1

        res.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)

        res.layer4[0].conv1.weight.data = weight1_1

        res.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)

        res.layer2[0].conv1.weight.data = weight3_1



        #print(res)
        # print(res.layer4[0].conv2)

        # res.conv1.weight[:, :3] = weight
        # res.conv1.weight[:, 3:6] = weight
        # res.conv1.weight[:, 3:6] = res.conv1.weight[:, 0]

        self.net = nn.Sequential(*list(res.children())[:-2])

    def forward(self, x):
        res = []
        for layer in self.net:
            x = layer(x)
            res.append(x)
        # x = self.net(x)
        #     print("x", x.size())

        return x, res



class Res18(nn.Module):
    def __init__(self, pretrained=True, number_of_instances = 1):
        super(Res18, self).__init__()

        res = torchvision.models.resnet18(pretrained=True, replace_stride_with_dilation=[False, False, False])

        #res = FrozenBatchNorm.convert_frozen_batchnorm(res)
        #print(res)

        weight = res.conv1.weight.data

        weight1 = res.layer4[0].downsample[0].weight.data
        weight2 = res.layer3[0].downsample[0].weight.data
        weight3 = res.layer2[0].downsample[0].weight.data
        weight1_1 = res.layer4[0].conv1.weight.data
        weight2_1 = res.layer3[0].conv1.weight.data
        weight3_1 = res.layer2[0].conv1.weight.data

        res.conv1 = nn.Conv2d(3 + number_of_instances, 64, kernel_size=7, stride=2, padding=3, bias=False)

        res.conv1.weight[:, :3].data = weight

        # res.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        #
        # res.layer4[0].conv2.weight.data = weight1
        #
        res.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)

        res.layer3[0].downsample[0].weight.data = weight2

        res.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)

        res.layer4[0].downsample[0].weight.data = weight1

        res.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)

        res.layer2[0].downsample[0].weight.data = weight3

        res.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)

        res.layer3[0].conv1.weight.data = weight2_1

        res.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)

        res.layer4[0].conv1.weight.data = weight1_1

        res.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)

        res.layer2[0].conv1.weight.data = weight3_1



        #print(res)
        # print(res.layer4[0].conv2)

        # res.conv1.weight[:, :3] = weight
        # res.conv1.weight[:, 3:6] = weight
        # res.conv1.weight[:, 3:6] = res.conv1.weight[:, 0]

        self.net = nn.Sequential(*list(res.children())[:-2])

    def forward(self, x):
        res = []
        for layer in self.net:
            x = layer(x)
            res.append(x)
        # x = self.net(x)
        #print("x", x.size())

        return x, res






# def res50():
#
#     res50 = Res34(pretrained=True)
#
#     #res50 = nn.Sequential(*list(res50.children())[1:-1])
#
#     print(res50)
#
#
# ucf = res50()