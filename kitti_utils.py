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

    def projection_velo(self, points_3d_rect):

        points_3d = (np.dot(np.linalg.inv(self.R0), points_3d_rect.T)).T
        points_3d= np.hstack((points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)))
        points_3d = np.dot(points_3d, self.cam_vel.T)
        return points_3d

    def corners3d_bboxes(self, corners3d):
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        points = np.matmul(corners3d_hom, self.P.T)  # (N, 8, 3)

        x, y = points[:, :, 0] / points[:, :, 2], points[:, :, 1] / points[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    
def compute_box_3d(obj, proj):

    Ry = np.array([[np.cos(obj.ry),  0,  np.sin(obj.ry)],
                    [0,  1,  0],
                    [-(np.sin(obj.ry)), 0,  np.cos(obj.ry)]])

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    
    # corners of bbox 3d
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        
    corn_3d = np.dot(Ry, np.vstack([x_corners,y_corners,z_corners]))
    corn_3d[0,:] = corn_3d[0,:] + obj.t[0]
    corn_3d[1,:] = corn_3d[1,:] + obj.t[1]
    corn_3d[2,:] = corn_3d[2,:] + obj.t[2]
    # 3d bounding box for objs in front of the camera
    if np.any(corn_3d[2,:] < 0.1):
        corn_2d = None
        return corn_2d, corn_3d.T


    pts_3d = np.hstack((corn_3d.T, np.ones(((corn_3d.T).shape[0],1))))
    pts_2d = np.dot(pts_3d, proj.T)
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    corn_2d = pts_2d[:,0:2]
    return corn_2d, corn_3d.T

def compute_orientation_3d(obj, proj):
    Ry = np.array([[np.cos(obj.ry),  0,  np.sin(obj.ry)],
                    [0,  1,  0],
                    [-(np.sin(obj.ry)), 0,  np.cos(obj.ry)]])

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l],[0,0],[0,0]])
    
    # rotation
    orientation_3d = np.dot(Ry, orientation_3d)
    # translation
    orientation_3d[0,:] = orientation_3d[0,:] + obj.t[0]
    orientation_3d[1,:] = orientation_3d[1,:] + obj.t[1]
    orientation_3d[2,:] = orientation_3d[2,:] + obj.t[2]
    
    if np.any(orientation_3d[2,:]<0.1):
        orientation_2d = None
        return orientation_2d, orientation_3d.T
    
    pts_3d_extend = np.hstack((orientation_3d.T, np.ones(((orientation_3d.T).shape[0],1))))
    pts_2d = np.dot(pts_3d_extend, proj.T) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    orientation_2d = pts_2d[:,0:2]
    return orientation_2d, orientation_3d.T