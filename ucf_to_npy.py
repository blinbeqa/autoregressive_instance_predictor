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


def ucf_to_npy():
    DATA_DIR = os.path.join("/home/beqa/PycharmProjects/conv-tt-lstm/conv-tt-lstm-master/code/datasets",
                              "UCF-101")
    annot_path = os.path.join("/home/beqa/PycharmProjects/conv-tt-lstm/conv-tt-lstm-master/code/datasets",
                              "ucfTrainTestlist")

    print("hini n ucf")
    print(DATA_DIR)

    data = UCF101(root=DATA_DIR, annotation_path=annot_path, frames_per_clip=11, train=True, num_workers=0)

    print("shape npy")
    print(data)

    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size

    train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
    print("Train Size")
    print(train_size)
    print("Validate Size")
    print(valid_size)

    # dataloaer for the training set

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                                    shuffle=False, num_workers=1, drop_last=True)

    train_size = len(train_data_loader) * 1

    # dataloaer for the valiation set

    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=1,
                                                    shuffle=False, num_workers=1, drop_last=True)

    valid_size = len(valid_data_loader) * 1

    print("train_data")
    #print(train_data.data)

    outputs = [None] * 5000

    counter1 = 0

    outputs = torch.stack([t[0].squeeze(0) for t in train_data_loader], dim=0)
    np.save('datasets/moving-mnist/ucf_temp.npy', outputs.numpy())

    # for frames in train_data_loader:
    #     print("frames")
    #     print(counter1)
    #     print(frames[0].size())
    #
    #     counter1 += 1
    #
    # for framesv in valid_data_loader:
    #     print("framesv")
    #     print(framesv[0].size())

    return valid_size

ucf = ucf_to_npy()