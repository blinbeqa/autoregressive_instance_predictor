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


class Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50, self).__init__()

        res = torchvision.models.resnet50(pretrained=True)

        # print(res)

        weight = res.conv1.weight.data
        weight1 = res.layer4[0].conv2.weight.data
        weight2 = res.layer4[0].downsample[0].weight.data

        res.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        res.conv1.weight[:, :3].data = weight

        res.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        res.layer4[0].conv2.weight.data = weight1

        res.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, dilation=2, bias=False)

        res.layer4[0].downsample[0].weight.data = weight2


        # print(res)
        # print(res.layer4[0].conv2)

        # res.conv1.weight[:, :3] = weight
        # res.conv1.weight[:, 3:6] = weight
        #res.conv1.weight[:, 3:6] = res.conv1.weight[:, 0]

        self.net = nn.Sequential(*list(res.children())[:-5])

    def forward(self, x):
        x = self.net(x)

        return x






# def res50():

#     res50 = Res50(pretrained=True)

#     #res50 = nn.Sequential(*list(res50.children())[1:-1])

#     print(res50)


# ucf = res50()