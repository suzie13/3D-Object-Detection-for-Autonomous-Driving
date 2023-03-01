import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import cv2
import math


object_list = {'Car':0, 'Van':1 , 'Truck':2 , 'Pedestrian':3 , 'Person_sitting':4 , 'Cyclist':5 , 'Tram':6 }
class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram' ]

bc={}
bc['minX'] = 0; bc['maxX'] = 50; bc['minY'] = -25; bc['maxY'] = 25; bc['minZ'] = -2.7; bc['maxZ'] = 1.3

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)

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

def camcoord_lidarcoord(boxes, V2C=None, R0=None, P2=None):
    result = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camsensor_lidarsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi/2
        result.append([x, y, z, h, w, l, rz])
    result = np.array(result).reshape(-1, 7)
    return result



def lidarcoord_camcoord(boxes,V2C=None, R0=None, P2=None):
    result = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidarsensor_camsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi/2
        result.append([x, y, z, h, w, l, ry])
    result = np.array(result).reshape(-1, 7)
    return result

def cam_lidar_bbox(box, V2C=None, R0=None, P2=None):
    result = []
    for b in box:
       x, y, z, h, w, l, ry = b
       (x, y, z), h, w, l, rz = camsensor_lidarsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi/2
       result.append([x, y, z, h, w, l, rz])
    result = np.array(result).reshape(-1, 7)
    return result

def lidar_camera_bbox(box,V2C=None, R0=None, P2=None):
	result = []
	for b in box:
		x, y, z, h, w, l, rz = b
		(x, y, z), h, w, l, ry = lidarsensor_camsensor(x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi/2
		result.append([x, y, z, h, w, l, ry])
	result = np.array(result).reshape(-1, 7)
	return result

