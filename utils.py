from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Polygon

import numpy as np
import math
import torch
import cv2
from utils import *
from math import cos, sin


object_list = {'Car': 0,'Pedestrian': 1,'Cyclist': 2,'Van': 0,'Person_sitting': 1}

boundary = {"minX": 0, "maxX": 50,"minY": -25,"maxY": 25, "minZ": -2.73,"maxZ": 1.27}

boundary_back = {"minX": -50,"maxX": 0,"minY": -25,"maxY": 25,"minZ": -2.73,"maxZ": 1.27}

DISCRETIZATION = (boundary["maxX"] - boundary["minX"])/608
colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]

Tr_velo_to_cam = np.array([
		[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
		[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
		[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
		[0, 0, 0, 1]
	])

# cal mean from train set
R0 = np.array([
		[0.99992475, 0.00975976, -0.00734152, 0],
		[-0.0097913, 0.99994262, -0.00430371, 0],
		[0.00729911, 0.0043753, 0.99996319, 0],
		[0, 0, 0, 1]
])

P2 = np.array([[719.787081,         0., 608.463003,    44.9538775],
               [        0., 719.787081, 174.545111,     0.1066855],
               [        0.,         0.,         1., 3.0106472e-03],
			   [0., 0., 0., 0]
])

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def rotated_box_wh_iou_polygon(anchor, wh, imre):
    w1, h1, im1, re1 = anchor[0], anchor[1], anchor[2], anchor[3]

    wh = wh.t()
    imre = imre.t()
    w2, h2, im2, re2 = wh[0], wh[1], imre[0], imre[1]

    anchor_box = torch.cuda.FloatTensor([100, 100, w1, h1, im1, re1]).view(-1, 6)    
    target_boxes = torch.cuda.FloatTensor(w2.shape[0], 6).fill_(100)

    target_boxes[:, 2] = w2
    target_boxes[:, 3] = h2
    target_boxes[:, 4] = im2
    target_boxes[:, 5] = re2

    ious = rotated_bbox_iou_polygon(anchor_box[0], target_boxes)

    return torch.from_numpy(ious)


def rotated_bbox_iou_polygon(box1, box2):
    box1 = ((box1).detach().cpu()).numpy()
    box2 = ((box2).detach().cpu()).numpy()

    x,y,w,l,im,re = box1
    angle = np.arctan2(im, re)
    bbox1 = np.array(get_corners(x, y, w, l, angle)).reshape(-1,4,2)

    bbox1 = np.array( [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in bbox1])

    bbox2 = []
    for i in range(box2.shape[0]):
        x,y,w,l,im,re = box2[i,:]
        angle = np.arctan2(im, re)
        bev_corners = get_corners(x, y, w, l, angle)
        bbox2.append(bev_corners)
    bbox2 = np.array( [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in np.array(bbox2)])



    iou = [(bbox1[0]).intersection(b).area / ((bbox1[0]).union(b).area + 1e-12) for b in bbox2]
    iou = np.array(iou, dtype=np.float32)

    return iou

def non_max_suppression_rotated_bbox(prediction, conf_thres=0.95, nms_thres=0.4):

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        #keep images only above a confidence threshold 
        image_pred = image_pred[image_pred[:, 6] >= conf_thres]
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 6] * image_pred[:, 7:].max(1)[0]
        # Sorting
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 7:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :7].float(), class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = rotated_bbox_iou_polygon(detections[0, :6], detections[:, :6]) > nms_thres
            large_overlap = torch.from_numpy(large_overlap)
            label_match = detections[0, -1] == detections[:, -1]

            invalid = large_overlap & label_match
            weights = detections[invalid, 6:7]
            # Merge overlapping bboxes by order of confidence
            detections[0, :6] = (weights * detections[invalid, :6]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.ByteTensor

    obj_mask = ByteTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    noobj_mask = ByteTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(1)
    class_mask = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    iou_scores = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    tx = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    ty = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    tw = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    th = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    tim = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    tre = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2)).fill_(0)
    tcls = FloatTensor(pred_boxes.shape[0], pred_boxes.size(1), pred_boxes.size(2), pred_boxes.size(2), pred_cls.size(-1)).fill_(0)

    t_bbox = target[:, 2:8]
    
    gxy = t_bbox[:, :2] * pred_boxes.size(2)
    gwh = t_bbox[:, 2:4] * pred_boxes.size(2)
    gimre = t_bbox[:, 4:]

    # Get anchors with best iou
    ious = torch.stack([rotated_box_wh_iou_polygon(a, gwh, gimre) for a in anchors])    

    _, best_n = ious.max(0)
    b, target_labels = target[:, :2].long().t()
    
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gim, gre = gimre.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    tim[b, best_n, gj, gi] = gim
    tre[b, best_n, gj, gi] = gre

    # One-hot encoding
    tcls[b, best_n, gj, gi, target_labels] = 1
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    box1_new = torch.cuda.FloatTensor((pred_boxes[b, best_n, gj, gi]).shape[0], 6).fill_(0)
    box2_new = torch.cuda.FloatTensor(t_bbox.shape[0], 6).fill_(0)

    box1_new[:, :4] = (pred_boxes[b, best_n, gj, gi])[:, :4]
    box1_new[:, 4:] = (pred_boxes[b, best_n, gj, gi])[:, 4:]

    box2_new[:, :4] = t_bbox[:, :4] * pred_boxes.size(2)
    box2_new[:, 4:] = t_bbox[:, 4:]

    ious = []
    for i in range(box1_new.shape[0]):
        bbox1 = box1_new[i]
        bbox2 = box2_new[i].view(-1, 6)

        iou = rotated_bbox_iou_polygon(bbox1, bbox2).squeeze()
        ious.append(iou) 

    ious = np.array(ious)

    ious = torch.from_numpy(ious)
    iou_scores[b, best_n, gj, gi] = ious.to('cuda:0')
     
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf


def removePoints(PointCloud, BoundaryCond):
    """Remove outlier point cloud points"""
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
            PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] - minZ

    return PointCloud

def makeBVFeature(PointCloud_, Discretization, bc):
    """Point Cloud processing to get bird's eye view feature map (RGB image)"""
    Height = 608 + 1
    Width = 608 + 1

    # Discretization
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    #sorting in x, y (x and y sorting in increasing) and z (z sorting in decreasing i.e top to bottom)
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map -> represented by green channel
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    max_height = float(np.abs(bc['maxZ'] - bc['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

    # Intensity Map ->represented by blue channel
    intensityMap = np.zeros((Height, Width))
    # Density Map -> represented by red channel
    densityMap = np.zeros((Height, Width))

    _, indices, cnt = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_cnt=True)
    pc = PointCloud[indices]

    normalized_cnt = np.minimum(1.0, np.log(cnt + 1) / np.log(64))

    intensityMap[np.int_(pc[:, 0]), np.int_(pc[:, 1])] = pc[:, 3]
    densityMap[np.int_(pc[:, 0]), np.int_(pc[:, 1])] = normalized_cnt

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:608, :608]  # r_map
    RGB_Map[1, :, :] = heightMap[:608, :608]  # g_map
    RGB_Map[0, :, :] = intensityMap[:608, :608]  # b_map

    return RGB_Map

def bev_labels(objects):
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox = []
            bbox.append(obj.cls_id)
            bbox.extend([obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry])
            bbox_selected.append(bbox)
    
    if (len(bbox_selected) == 0):
        return np.zeros((1, 8), dtype=np.float32), True
    else:
        bbox_selected = np.array(bbox_selected).astype(np.float32)
        return bbox_selected, False

def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)

    # front left
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    # rear left
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # rear right
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # front right
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    return bev_corners

def build_target(labels):
    bc = {"minX": 0,"maxX": 50,"minY": -25,"maxY": 25,"minZ": -2.73,"maxZ": 1.27}
    target = np.zeros([50, 7], dtype=np.float32)
    
    index = 0
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]

        l = l + 0.3
        w = w + 0.3

        yaw = np.pi * 2 - yaw
        if (x > bc["minX"]) and (x < bc["maxX"]) and (y > bc["minY"]) and (y < bc["maxY"]):
            y1 = (y - bc["minY"])/(bc["maxY"]-bc["minY"])
            x1 = (x - bc["minX"]) / (bc["maxX"]-bc["minX"])
            w1 = w/(bc["maxY"] - bc["minY"])
            l1 = l/(bc["maxX"] - bc["minX"])

            target[index][0] = cl
            target[index][1] = y1 
            target[index][2] = x1
            target[index][3] = w1
            target[index][4] = l1
            target[index][5] = math.sin(float(yaw))
            target[index][6] = math.cos(float(yaw))

            index = index+1

    return target

def camsensor_lidarsensor(x, y, z, V2C=None,R0=None,P2=None):
	val = np.array([x, y, z, 1])
	if V2C is None or R0 is None:
		val = np.matmul(R0_inv, val)
		val = np.matmul(Tr_velo_to_cam_inv, val)
	else:
		R0_i = np.zeros((4,4))
		R0_i[:3,:3] = R0
		R0_i[3,3] = 1
		val = np.matmul(np.linalg.inv(R0_i), val)
		inv = np.zeros_like(V2C)
		inv[0:3,0:3] = np.transpose(V2C[0:3,0:3])
		inv[0:3,3] = np.dot(-np.transpose(V2C[0:3,0:3]), V2C[0:3,3])
		val = np.matmul(inv, val)
	val = val[0:3]
	val = tuple(val)
	return val

def lidarsensor_camsensor(x, y, z,V2C=None, R0=None, P2=None):
    """ 
    P2->projection matrix for 3D to 2D (4,4)
    R0->rotation matrix representing sensor orientation relative to world (4,4)
    V2C->velodyne to camera transformation matrix (4,4)
    """
	p = np.array([x, y, z, 1])
	if V2C is None or R0 is None:
        # point already in camera coordinates therefore Tr_vel_to_cam used to transform to camera coordinates
		p = np.matmul(Tr_velo_to_cam, p)
		p = np.matmul(R0, p)
	else:
        # transform p point from lidar to camera coordinates
		p = np.matmul(V2C, p)
		p = np.matmul(R0, p)
	p = p[0:3]
	return tuple(p)

def camcoord_lidarcoord(points):
    """coordinates from camera coordinate system to lidar coordinate system"""
	points = np.hstack([points, np.ones((len(points), 1))]).T

	points = np.matmul(R0_inv, points)
	points = np.matmul(Tr_velo_to_cam_inv, points).T 
	points = points[:, 0:3]
	res = points.reshape(-1, 3)
	return res

def lidarcoord_camcoord(points, V2C=None, R0=None):
    """coordinates from lidar coordinate system to camera coordinate system"""
	N = points.shape[0]
    #Rotation from lidar to camera
	R0 = np.array([
		[0.99992475, 0.00975976, -0.00734152, 0],
		[-0.0097913, 0.99994262, -0.00430371, 0],
		[0.00729911, 0.0043753, 0.99996319, 0],
		[0, 0, 0, 1]
    ])
	points = np.hstack([points, np.ones((N, 1))]).T

	if V2C is None or R0 is None:
		points = np.matmul(Tr_velo_to_cam, points)
		points = np.matmul(R0, points).T
	else:
		points = np.matmul(V2C, points)
		points = np.matmul(R0, points).T
	points = points[:, 0:3]
	res = points.reshape(-1, 3)
	return res

def cam_lidar_bbox(boxes, V2C=None, R0=None, P2=None):
    """ transformation of bounding box coordinates from camera frame to lidar frame"""
	ret = []
	for box in boxes:
		x, y, z, h, w, l, ry = box
		(x, y, z), h, w, l, rz = camsensor_lidarsensor(
			x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi / 2
		ret.append([x, y, z, h, w, l, rz])
	return np.array(ret).reshape(-1, 7)

def lidar_camera_bbox(boxes,V2C=None, R0=None, P2=None):
    """transformation of bounding box coordinates from lidar frame to sensor frame"""
	ret = []
	for box in boxes:
	x, y, z, h, w, l, rz = box
        #(x, y, z) -> transformed center coordinates of box in the camera coordinate system
		(x, y, z), h, w, l, ry = lidarsensor_camsensor(
			x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi / 2
		ret.append([x, y, z, h, w, l, ry])
	return np.array(ret).reshape(-1, 7)	


def center_to_corner_box3d(boxes_center, coordinate='lidar'):
	N = boxes_center.shape[0]
	ret = np.zeros((N, 8, 3), dtype=np.float32)

	if coordinate == 'camera':
		boxes_center = cam_lidar_bbox(boxes_center)

	for i in range(N):
		box = boxes_center[i]
		translation = box[0:3]
		size = box[3:6]
		rotation = [0, 0, box[-1]]

		h, w, l = size[0], size[1], size[2]
		trackletBox = np.array([ 
			[-l/ 2, -l/2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
			[w/2, -w/2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
			[0, 0, 0, 0, h, h, h, h]])

		yaw = rotation[2]
		rotMat = np.array([
			[np.cos(yaw), -np.sin(yaw), 0.0],
			[np.sin(yaw), np.cos(yaw), 0.0],
			[0.0, 0.0, 1.0]])
		cornerPosInVelo = np.dot(rotMat, trackletBox) + \
			np.tile(translation, (8, 1)).T
		box3d = cornerPosInVelo.transpose()
		ret[i] = box3d

	if coordinate == 'camera':
		for idx in range(len(ret)):
			ret[idx] = lidarcoord_camcoord(ret[idx])

	return ret

CORNER2CENTER_AVG = True
def corner_to_center_box3d(boxes_corner, coordinate='camera'):

	# (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry/z
	if coordinate == 'lidar':
		for idx in range(len(boxes_corner)):
			boxes_corner[idx] = lidarcoord_camcoord(boxes_corner[idx])

	ret = []
	for roi in boxes_corner:
		if CORNER2CENTER_AVG: 
			roi = np.array(roi)
			h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
			w, l, ry = 0, 0, 0
			for i, j in [(0, 3), (1, 2), (4, 7), (5, 6)]:
				w += np.sqrt(np.sum((roi[i, [0, 2]] - roi[j, [0, 2]])**2))
			w = w/4

			for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
				l += np.sqrt(np.sum((roi[i, [0, 2]] - roi[j, [0, 2]])**2))
			l = l/4

			x = np.sum(roi[:, 0], axis=0) / 8
			y = np.sum(roi[0:4, 1], axis=0) / 4
			z = np.sum(roi[:, 2], axis=0) / 8

			for i, j in [(2, 1), (6, 5), (3, 0), (7, 4)]:
				ry += math.atan2(roi[i, 0] - roi[j, 0], roi[i, 2] - roi[j, 2])
			for i, j in [(0, 1), (4, 5), (3, 2), (7, 6)]:
				ry += math.atan2(roi[i, 2] - roi[j, 2], roi[j, 0] - roi[i, 0])
			ry = ry/8

			if w > l:
				w, l = l, w
				ry = ry - np.pi/2
			elif l > w:
				l, w = w, l
				ry = ry - np.pi/2
			ret.append([x, y, z, h, w, l, ry])
			
		else: 
			h = max(abs(roi[:4, 1] - roi[4:, 1]))
			w = np.max(
				np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
			)
			l = np.max(
				np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
			)
			x = np.sum(roi[:, 0], axis=0) / 8
			y = np.sum(roi[0:4, 1], axis=0) / 4
			z = np.sum(roi[:, 2], axis=0) / 8
			ry = np.sum(
				math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
				math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
				math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
				math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
				math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
				math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
				math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
				math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
			) / 8
			if w > l:
				w, l = l, w

				deg = 5
				ang = ry + np.pi/2
				while ang >= np.pi/2:
					ang -= np.pi
				while ang < -np.pi/2:
					ang += np.pi
				if abs(ang + np.pi/2) < deg/180 * np.pi:
					ang = np.pi/2
				ry = ang
			ret.append([x, y, z, h, w, l, ry])
	
	if coordinate == 'lidar':
		ret = lidar_camera_bbox(np.array(ret))

	return np.array(ret)

def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):

	N = points.shape[0]
	points = np.hstack([points, np.ones((N, 1))])

	mat1 = np.eye(4)
	mat1[3, 0:3] = tx, ty, tz
	points = np.matmul(points, mat1)
	mat = np.zeros((4, 4))

	if rx != 0:
		mat = np.array([[1, 0, 0, 0],
						[0, cos(rx), -sin(rx), 0],
						[0, sin(rx), cos(rx), 0],
						[0, 0, 0, 1]])
		points = np.matmul(points, mat)

	if ry != 0:
		mat = np.array([[cos(ry), 0, sin(ry), 0],
						[0, 1, 0, 0],
						[-sin(ry), 0, cos(ry), 0],
						[0, 0, 0, 1]])
		points = np.matmul(points, mat)


	if rz != 0:
		mat = np.array([[cos(rz), -sin(rz), 0, 0],
						[sin(rz), cos(rz), 0, 0],
						[0, 0, 1, 0],
						[0, 0, 0, 1]])
		points = np.matmul(points, mat)

	return points[:, 0:3]
