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