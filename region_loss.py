from __future__ import division
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils import *



def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    if x.is_cuda:
        self.mse_loss = nn.MSELoss(size_average=True).cuda()
        self.bce_loss = nn.BCELoss(size_average=True).cuda()
        self.ce_loss = nn.CrossEntropyLoss().cuda()
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        ByteTensor = torch.cuda.ByteTensor

    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor
        ByteTensor = torch.ByteTensor


    # ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    # FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.ByteTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    mask = torch.zeros(ByteTensor(nB, nA, nG, nG))
    conf_mask = torch.ones(ByteTensor(nB, nA, nG, nG))
    cls_mask = torch.zeros(FloatTensor(nB, nA, nG, nG))
    iou = torch.zeros(FloatTensor(nB, nA, nG, nG))
    tx =torch.zeros( FloatTensor(nB, nA, nG, nG))
    ty = torch.zeros(FloatTensor(nB, nA, nG, nG))
    tw = torch.zeros(FloatTensor(nB, nA, nG, nG))
    th = torch.zeros(FloatTensor(nB, nA, nG, nG))
    tim = torch.zeros(FloatTensor(nB, nA, nG, nG))
    tre = torch.zeros(FloatTensor(nB, nA, nG, nG))
    tcls = torch.zeros(FloatTensor(nB, nA, nG, nG, nC))

    # Convert to position relative to box
    target_boxes = target[:, 2:8]
    
    gxy = target_boxes[:, :2] * nG
    gwh = target_boxes[:, 2:4] * nG
    gimre = target_boxes[:, 4:]

    # Get anchors with best iou
    ious = torch.stack([rotated_box_wh_iou_polygon(anchor, gwh, gimre) for anchor in anchors])    

    best_ious, best_n = ious.max(0)
    b, target_labels = target[:, :2].long().t()
    
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gim, gre = gimre.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # Im and real part 
    tim[b, best_n, gj, gi] = gim
    tre[b, best_n, gj, gi] = gre

    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    rotated_iou_scores = rotated_box_11_iou_polygon(pred_boxes[b, best_n, gj, gi], target_boxes, nG)
    iou_scores[b, best_n, gj, gi] = rotated_iou_scores.to('cuda:0')
     
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf