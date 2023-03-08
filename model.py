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
            n_classes = int(c["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            layer = Layers(anchors, n_classes, img_size)
            modules.add_module(f"yolo_{i}", layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list



class Layers(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(Layers, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def forward(self, x, targets=None, img_dim=None):

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        imagin = prediction[..., 4]
        real = prediction[..., 5] 
        pred_conf = torch.sigmoid(prediction[..., 6])  # predicted comfidence
        pred_cls = torch.sigmoid(prediction[..., 7:])  # class confidence

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            # self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
            # compute_grid_offsets(self, grid_size, cuda=True):
            self.grid_size = grid_size
            g = self.grid_size
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            self.stride = self.img_dim / self.grid_size
            # Calculate offsets for each grid
            self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
            self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
            self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride, im, re) for a_w, a_h, im, re in self.anchors])
            self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
            self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        bboxes = FloatTensor(prediction[..., :6].shape)
        bboxes[..., 0] = x.data + self.grid_x
        bboxes[..., 1] = y.data + self.grid_y
        bboxes[..., 2] = torch.exp(w.data) * self.anchor_w
        bboxes[..., 3] = torch.exp(h.data) * self.anchor_h
        bboxes[..., 4] = imagin
        bboxes[..., 5] = real

        output = torch.cat(
            (
                bboxes[..., :4].view(num_samples, -1, 4) * self.stride,
                bboxes[..., 4:].view(num_samples, -1, 2),
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )


class COMPLEXYOLO(nn.Module):

    def __init__(self, config_path, img_size=416):
        super(COMPLEXYOLO, self).__init__()

        file = open(config_path, 'r')
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines] 
        config = []
        for line in lines:
            if line.startswith('['): 
                config.append({})
                config[-1]['type'] = line[1:-1].rstrip()
                if config[-1]['type'] == 'convolutional':
                    config[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                config[-1][key.rstrip()] = value.strip()
        self.config = config
        self.hyperparams, self.module_list = create_modules(self.config)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_output =  []
        yolo_output = []
        for i, (c, module) in enumerate(zip(self.config, self.module_list)):
            if c["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif c["type"] == "route":
                x = torch.cat([layer_output[int(layer_i)] for layer_i in c["layers"].split(",")], 1)
            elif c["type"] == "shortcut":
                layer_i = int(c["from"])
                x = layer_output[-1] + layer_output[layer_i]
            elif c["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_output.append(x)
            layer_output.append(x)
        yolo_output = (torch.cat(yolo_output, 1)).detach().cpu()
        return yolo_output if targets is None else (loss, yolo_output)
