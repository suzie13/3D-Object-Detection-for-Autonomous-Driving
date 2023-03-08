from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

from utils.utils import *

def create_modules(config):

    hyperparams = config.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, c in enumerate(config):
        modules = nn.Sequential()

        if c["type"] == "convolutional":
            bn = int(c["batch_normalize"])
            filters = int(c["filters"])
            kernel_size = int(c["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(f"conv_{i}",
                nn.Conv2d(in_channels=output_filters[-1],out_channels=filters, kernel_size=kernel_size, stride=int(c["stride"]),padding=pad,bias=not bn,),)
            if bn:
                modules.add_module(f"batch_norm_{i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if c["activation"] == "leaky":
                modules.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))

        elif c["type"] == "maxpool":
            kernel_size = int(c["size"])
            stride = int(c["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{i}", maxpool)

        elif c["type"] == "upsample":
            modules.add_module(
            f"upsample_{i}",nn.Upsample(scale_factor=int(c["stride"]), mode="nearest"))

        elif c["type"] == "route":
            layers = [int(x) for x in c["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{i}", nn.Identity())

        elif c["type"] == "shortcut":
            filters = output_filters[1:][int(c["from"])]
            modules.add_module(f"shortcut_{i}", nn.Identity())

        elif c["type"] == "yolo":
            anchor_index = [int(x) for x in c["mask"].split(",")]
            # get anchor
            anchors = [float(x) for x in c["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1], math.sin(anchors[i + 2]), math.cos(anchors[i + 2])) for i in range(0, len(anchors), 3)]
            anchors = [anchors[i] for i in anchor_index]

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list