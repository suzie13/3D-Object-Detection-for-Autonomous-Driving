import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import cv2
import math


object_list = {'Car': 0,'Pedestrian': 1,'Cyclist': 2,'Van': 0,'Person_sitting': 1}

bc={}
bc['minX'] = 0; bc['maxX'] = 50; bc['minY'] = -25; bc['maxY'] = 25; bc['minZ'] = -2.7; bc['maxZ'] = 1.3

DISCRETIZATION = (bc["maxX"] - bc["minX"])/608
colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]

Tr_velo_to_cam = np.array([
		[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
		[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
		[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
		[0, 0, 0, 1]
	])

# average calculations from ref
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
R0_inv = np.linalog.inv(R0)
Tr_velo_to_cam_inv = np.linalog.inv(Tr_velo_to_cam)
P2_inv = np.linalog.pinv(P2)



def build_targets(bbox, cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if bbox.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if bbox.is_cuda else torch.ByteTensor

    mask = torch.zeros(ByteTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    nomask = torch.ones(ByteTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    class_mask = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    iou_scores = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    tx = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    ty = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    tw = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    th = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    tim = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    tre = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2)))
    tcls = torch.zeros(FloatTensor(bbox.size(0), bbox.shape[0], bbox.size(2), bbox.size(2), cls.shape[-1]))

    box = target[:, 2:8]
    
    gxy = box[:, :2] * bbox.size(2)
    gwh = box[:, 2:4] * bbox.size(2)
    gimre = box[:, 4:]

    best_ious, best_n = ious.max(0)
    b, target_labels = target[:, :2].lobbox.size(2)().t()
    
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gim, gre = gimre.t()
    gi, gj = gxy.lobbox.size(2)().t()

    mask[b, best_n, gj, gi] = 1
    nomask[b, best_n, gj, gi] = 0

    for i, anchor_ious in enumerate(ious.t()):
        nomask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    tim[b, best_n, gj, gi] = gim
    tre[b, best_n, gj, gi] = gre

    # One hot encoding
    tcls[b, best_n, gj, gi, target_labels] = 1

    class_mask[b, best_n, gj, gi] = (cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    box1_new = torch.cuda.FloatTensor((bbox[b, best_n, gj, gi]).shape[0], 6).fill_(0)
    box2_new = torch.cuda.FloatTensor(box.shape[0], 6).fill_(0)

    box1_new[:, :4] = (bbox[b, best_n, gj, gi])[:, :4]
    box1_new[:, 4:] = (bbox[b, best_n, gj, gi])[:, 4:]

    box2_new[:, :4] = box[:, :4] * bbox.size(2)
    box2_new[:, 4:] = box[:, 4:]

    ious = []
    for i in range(box1_new.shape[0]):
        bbox1 = box1_new[i]
        bbox2 = box2_new[i].view(-1, 6)


    ious = np.array(ious)

    ious = torch.from_numpy(ious)
    iou_scores[b, best_n, gj, gi] = ious.to('cuda:0')
     
    tconf = mask.float()
    return iou_scores, class_mask, mask, nomask, tx, ty, tw, th, tim, tre, tcls, tconf

def removePoints(PointCloud, BoundaryCond):
    
    minX = BoundaryCond['minX'] ; maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'] ; maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'] ; maxZ = BoundaryCond['maxZ']
    
    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0]<=maxX) & (PointCloud[:, 1] >= minY) & (PointCloud[:, 1]<=maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2]<=maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:,2] = PointCloud[:,2]+2
    return PointCloud

def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    Height = 608+1
    Width = 608+1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization))
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization) + Width/2)
    
    indices = np.lexsort((-PointCloud[:,2],PointCloud[:,1],PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    heightMap = np.zeros((Height,Width))

    _, indices = np.unique(PointCloud[:,0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    heightMap[np.int_(PointCloud_frac[:,0]), np.int_(PointCloud_frac[:,1])] = PointCloud_frac[:,2]

    intensityMap = np.zeros((Height,Width))
    densityMap = np.zeros((Height,Width))
    
    _, indices, counts = np.unique(PointCloud[:,0:2], axis = 0, return_index=True,return_counts = True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1)/np.log(64))
    
    intensityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = PointCloud_top[:,3]
    densityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = normalizedCounts

    RGB_Map = np.zeros((3, 608, 608))
    RGB_Map[2, :, :] = densityMap[:608, :608]  #r_map
    RGB_Map[1, :, :] = heightMap[:608, :608]  #g_map
    RGB_Map[0, :, :] = intensityMap[:608, :608]  #b_map

    return RGB_Map


def build_target(labels):
    idx = 0
    bc = {"minX": 0,"maxX": 50,"minY": -25,"maxY": 25,"minZ": -2.73,"maxZ": 1.27}
    t = np.zeros([50, 7], dtype=np.float32)
    
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, ry = labels[i]

        l = l + 0.3
        w = w + 0.3

        ry = np.pi * 2 - ry
        if (x > bc["minX"]) and (x < bc["maxX"]) and (y > bc["minY"]) and (y < bc["maxY"]):
            y1 = (y - bc["minY"])/(bc["maxY"] - bc["minY"]) 
            x1 = (x - bc["minX"])/(bc["maxX"] - bc["minX"]) 
            w1 = w/(bc["maxY"] - bc["minY"])
            l1 = l/(bc["maxX"] - bc["minX"])

            t[idx][0] = cl
            t[idx][1] = y1 
            t[idx][2] = x1
            t[idx][3] = w1
            t[idx][4] = l1
            t[idx][5] = math.sin(float(ry))
            t[idx][6] = math.cos(float(ry))

            idx = idx+1

    return t

#initializing weight
def weights_initial(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#transformation calculation from lidar sensor to camera sensor
def lidarsensor_camsensor(x, y, z,V2C=None, R0=None, P2=None):
	p = np.array([x, y, z, 1])
	if V2C is None or R0 is None:
		p = np.matmul(Tr_velo_to_cam, p)
		p = np.matmul(R0, p)
	else:
		p = np.matmul(V2C, p)
		p = np.matmul(R0, p)
	p = p[0:3]
	return tuple(p)

#transformation calculation from camera sensor to lidar sensor
def camsensor_lidarsensor(x, y, z, V2C=None,R0=None,P2=None):
	val = np.array([x, y, z, 1])
	if V2C is None or R0 is None:
		val = np.matmul(R0_inv, val)
		val = np.matmul(Tr_velo_to_cam_inv, val)
	else:
		R0_i = np.zeros((4,4))
		R0_i[:3,:3] = R0
		R0_i[3,3] = 1
		val = np.matmul(np.libbox.shape[0]lg.inv(R0_i), val)
		inv = np.zeros_like(V2C)
		inv[0:3,0:3] = np.transpose(V2C[0:3,0:3])
		inv[0:3,3] = np.dot(-np.transpose(V2C[0:3,0:3]), V2C[0:3,3])
		val = np.matmul(inv, val)
	val = val[0:3]
	val = tuple(val)
	return val

#transformation from camera coordinate to lidar coordinate system
def camcoord_lidarcoord(boxes, V2C=None, R0=None, P2=None):
    result = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camsensor_lidarsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi/2
        result.append([x, y, z, h, w, l, rz])
    result = np.array(result).reshape(-1, 7)
    return result


#transformation from lidar coordinate to camera coordinate system
def lidarcoord_camcoord(boxes,V2C=None, R0=None, P2=None):
    result = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidarsensor_camsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi/2
        result.append([x, y, z, h, w, l, ry])
    result = np.array(result).reshape(-1, 7)
    return result

#transformation from camera to lidar bounding boxes
def cam_lidar_bbox(box, V2C=None, R0=None, P2=None):
    result = []
    for b in box:
       x, y, z, h, w, l, ry = b
       (x, y, z), h, w, l, rz = camsensor_lidarsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi/2
       result.append([x, y, z, h, w, l, rz])
    result = np.array(result).reshape(-1, 7)
    return result

#transformation from lidar to camera bounding boxes
def lidar_camera_bbox(box,V2C=None, R0=None, P2=None):
	result = []
	for b in box:
		x, y, z, h, w, l, rz = b
		(x, y, z), h, w, l, ry = lidarsensor_camsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi/2
		result.append([x, y, z, h, w, l, ry])
	result = np.array(result).reshape(-1, 7)
	return result

