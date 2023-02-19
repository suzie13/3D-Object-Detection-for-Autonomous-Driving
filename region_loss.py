from __future__ import division
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils import *


class RegionLoss(nn.Module):
    def __init__(self):
        super(RegionLoss, self).__init__()

        self.anchors = anchors
        self.num_anchors = 5
        self.bbox_attrs = 7+7
        self.thresh = 0.6
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss



    def forward(self, x, targets):

        batches = x.data.size(0)  # batch_size

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




        # print("prediction=",x.view(batches,5, self.bbox_attrs, 16, 32).permute(0, 1, 3, 4, 2).contiguous().shape)
        prediction = x.view(batches, self.num_anchors, self.bbox_attrs, 16, 32).permute(0, 1, 3, 4, 2).contiguous()  # prediction [12,5,16,32,15]
        
        # print("prediction shape=", prediction.shape)#prediction shape= torch.Size([12, 5, 8, 16, 15])
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  
        h = prediction[..., 3]  


        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        grid_x = torch.arange(32).repeat(16, 1).view([1, 1, 16, 32]).type(FloatTensor)
        grid_y = torch.arange(16).repeat(32, 1).t().view([1, 1, 16, 32]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w , a_h ) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, 5, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, 5, 1, 1))

        # scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h


        num_batches = targets.cpu().data.size(0)

        #initializing
        mask = torch.zeros(num_batches,5,16,32)
        conf_mask  = torch.ones(num_batches, 5, 16, 32)
        tx         = torch.zeros(num_batches, 5, 16, 32)
        ty         = torch.zeros(num_batches, 5, 16, 32) 
        tw         = torch.zeros(num_batches, 5, 16, 32) 
        th         = torch.zeros(num_batches, 5, 16, 32)
        tconf      = torch.zeros(num_batches, 5, 16, 32)
        tcls       = torch.zeros(num_batches, 5, 16, 32 , 7)


        for i in range(batches):
            for t in range(targets.cpu().data.shape[1]):
                if targets.cpu().data[i][t].sum() == 0:
                    continue

                # position relative to box
                gx = targets.cpu().data[i, t, 1] * 32
                gy = targets.cpu().data[i, t, 2] * 16
                gw = targets.cpu().data[i, t, 3] * 32
                gh = targets.cpu().data[i, t, 4] * 16

                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
                iou = bbox_iou(gt_box, anchor_shapes)
                gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

                indexes = torch.where(iou > self.thresh)
                conf_mask[i, indexes[0], int(gy), int(gx)] = 0
                best_iou = np.argmax(iou)
                mask[i, best_iou, int(gy), int(gx)] = 1
                conf_mask[i, best_iou, int(gy), int(gx)] = 1

                tx[i, best_iou, int(gy), int(gx)] = gx - int(gx)
                ty[i, best_iou, int(gy), int(gx)] = gy - int(gy)

                tw[i, best_iou, int(gy), int(gx)] = math.log(gw / anchors[best_iou][0] + 1e-16)
                th[i, best_iou, int(gy), int(gx)] = math.log(gh / anchors[best_iou][1] + 1e-16)
                # One-hot encoding
                target_label = int(targets.cpu().data[i, t, 0])
                tcls[i, best_iou, int(gy), int(gx), target_label] = 1
                tconf[i, best_iou, int(gy), int(gx)] = 1


        mask = Variable(mask.type(ByteTensor))
        conf_mask = Variable(conf_mask.type(ByteTensor))

        tx = Variable(tx.type(FloatTensor), requires_grad=False)
        ty = Variable(ty.type(FloatTensor), requires_grad=False)
        tw = Variable(tw.type(FloatTensor), requires_grad=False)
        th = Variable(th.type(FloatTensor), requires_grad=False)
        tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
        tcls = Variable(tcls.type(LongTensor), requires_grad=False)

        # Mask outputs to ignore non-existing objects
        loss_x = self.mse_loss(x[mask], tx[mask])
        loss_y = self.mse_loss(y[mask], ty[mask])
        loss_w = self.mse_loss(w[mask], tw[mask])
        loss_h = self.mse_loss(h[mask], th[mask])
        loss_conf = self.bce_loss(pred_conf[conf_mask - mask], tconf[conf_mask - mask]) + self.bce_loss(
            pred_conf[mask], tconf[mask]
        )
        loss_cls = (1 / batches) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # print( recall %f, precision %f, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % \
        #          (recall,  precision , loss_x.data, loss_y.data, loss_w.data, loss_h.data, loss_conf.data, loss_cls.data,loss.data))

        return loss


def build_targets(pred_boxes,pred_conf, pred_cls, target, anchors, ignore_thres, pred_boxes_1):
    nB = target.size(0)
    nTrueBox = target.data.size(1)
    mask = torch.zeros(target.size(0),5,16,32)
    conf_mask  = torch.ones(target.size(0),5, 16, 32)
    coord_mask = torch.zeros(target.size(0),5, 16, 32)
    tx         = torch.zeros(target.size(0),5, 16, 32)
    ty         = torch.zeros(target.size(0),5, 16, 32) 
    tw         = torch.zeros(target.size(0),5, 16, 32) 
    tl         = torch.zeros(target.size(0),5, 16, 32)
    tim        = torch.zeros(target.size(0),5, 16, 32)
    tre        = torch.zeros(target.size(0),5, 16, 32)
    tconf      = torch.zeros(target.size(0),5, 16, 32)
    tcls       = torch.zeros(target.size(0),5, 16, 32 , 8)

    nAnchors =5*16*32
    for b in range(nB):
        cur_pred_boxes = pred_boxes_1[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(nTrueBox):
            if target[b][t][1] == 0:
                break
            gx = target[b][t][1]*32
            gy = target[b][t][2]*16
            gw = target[b][t][3]*32
            gl = target[b][t][4]*16
            gim= target[b][t][5]
            gre= target[b][t][6]
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gl]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask = conf_mask.view(target.size(0), nAnchors)
        conf_mask[b][cur_ious>ignore_thres] = 0
    conf_mask = conf_mask.view(target.size(0),5, 16, 32)
    nGT = 0
    nCorrect = 0
    for b in range(target.size(0)):
        for t in range(target.shape[1]):
            if target[b][t].sum() == 0:
                continue

            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * 32
            gy = target[b, t, 2] * 16
            gw = target[b, t, 3] * 32
            gl = target[b, t, 4] * 16
            gim = target[b][t][5]
            gre = target[b][t][6]

            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gl])).unsqueeze(0)
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)

            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gl])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1

            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            tl[b, best_n, gj, gi] = math.log(gl / anchors[best_n][1] + 1e-16)

            tim[b][best_n][gj][gi]= target[b][t][5]
            tre[b][best_n][gj][gi]= target[b][t][6]
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # iou between ground truth and best prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, tl, tconf, tcls, tim, tre
