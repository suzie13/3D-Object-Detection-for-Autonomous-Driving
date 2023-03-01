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

