from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from util import count_parameters as count
from util import convert2cpu as cpu
from util import predict_transform


config_file = 'cfg/yolov4.cfg'
blocks = parse_cfg(config_file)
# darknet_details = blocks[0]
# channels = 3 
# #list of filter numbers in each layer.It is useful while defining number of filters in routing layer
# output_filters = []  
# modulelist = nn.ModuleList()

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    'linear': nn.Identity(),
    "mish": Mish()}

class MaxPoolStride1(nn.Module):
    def __init__(self, size=2):
        super(MaxPoolStride1, self).__init__()
        self.size = size
        if (self.size - 1) % 2 == 0:
            self.padding1 = (self.size - 1) // 2
            self.padding2 = self.padding1
        else:
            self.padding1 = (self.size - 1) // 2
            self.padding2 = self.padding1 + 1

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (self.padding1, self.padding2, self.padding1, self.padding2), mode='replicate'),
                         self.size, stride=1)
        return x

class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv9 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')