from __future__ import print_function

import numpy as np
import cv2
import os

class Class_3d(object):
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.type = data[0] # class name
        class_dict = {'Car': 0,'Van': 0, 'Truck':0, 'Tram': 0,'Pedestrian': 1,'Person_sitting': 1,'Cyclist': 2 , 'DontCare': -1, 'Misc': -1}
        self.cls_id = class_dict[self.type]

        self.truncation = data[1] 
        self.occlusion = int(data[2]) 
        self.alpha = data[3] # [-pi..pi]
        self.xmin = data[4]
        self.ymin = data[5]
        self.xmax = data[6]
        self.ymax = data[7]
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        self.h = data[8] # height
        self.w = data[9] # width
        self.l = data[10]
        self.t = (data[11],data[12],data[13]) 
        self.dis_to_cam = np.linalg.norm(self.t)
        self.ry = data[14] 
        self.score = data[15] if data.__len__() == 16 else -1.0


class Calibrate(object):
    def __init__(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
    
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        calib = {'P2': P2.reshape(3, 4),'P3': P3.reshape(3, 4),'R_rect': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
        self.proj = calib['P2'] 
        self.proj = np.reshape(self.P, [3,4])
        self.vel_cam = calib['Tr_velo2cam']
        self.vel_cam = np.reshape(self.vel_cam, [3,4])

        inv = np.zeros_like(self.vel_cam) 
        inv[0:3,0:3] = ((self.vel_cam)[0:3,0:3]).T
        inv[0:3,3] = np.dot((-((self.vel_cam)[0:3,0:3])).T, (self.vel_cam)[0:3,3])
        self.cam_vel = inv
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calib['R_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.proj[0,2]
        self.c_v = self.proj[1,2]
        self.f_u = self.proj[0,0]
        self.f_v = self.proj[1,1]
        self.b_x = self.proj[0,3]/(-self.f_u)
        self.b_y = self.proj[1,3]/(-self.f_v)
