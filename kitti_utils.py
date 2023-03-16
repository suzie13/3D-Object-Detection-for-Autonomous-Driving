from __future__ import print_function

import numpy as np
import cv2
import os

class Object3d(object):
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.type = data[0] # class 
        class_dict = {'Car': 0,'Van': 0, 'Truck':0, 'Tram': 0,'Pedestrian': 1,'Person_sitting': 1,'Cyclist': 2 , 'DontCare': -1, 'Misc': -1}
        self.cls_id = class_dict[self.type]

        self.truncation = data[1] 
        self.occlusion = int(data[2])
        self.alpha = data[3]

        self.xmin = data[4]
        self.ymin = data[5]
        self.xmax = data[6]
        self.ymax = data[7]
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

        self.h = data[8] # height
        self.w = data[9] # width
        self.l = data[10] # length
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.dis_to_cam = np.linalg.norm(self.t)
        self.ry = data[14] # yaw angle 
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
        self.P = calib['P2'] 
        self.P = np.reshape(self.P, [3,4])
        self.V2C = calib['Tr_velo2cam']
        self.V2C = np.reshape(self.V2C, [3,4])

        inv_Tr = np.zeros_like(self.V2C) # 3x4
        inv_Tr[0:3,0:3] = ((self.V2C)[0:3,0:3]).T
        inv_Tr[0:3,3] = np.dot((-((self.V2C)[0:3,0:3])).T, (self.V2C)[0:3,3])
        self.C2V = inv_Tr
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calib['R_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)


 
    def project_rect_to_velo(self, pts_3d_rect):

        pts_3d_ref = (np.dot(np.linalg.inv(self.R0), pts_3d_rect.T)).T
        pts_3d_ref= np.hstack((pts_3d_ref, np.ones((pts_3d_ref.shape[0], 1), dtype=np.float32)))
        pts_3d_ref = np.dot(pts_3d_ref, self.C2V.T)
        return pts_3d_ref

    def corners3d_to_img_boxes(self, corners3d):
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner


def compute_box_3d(obj, P):

    R_yaw = np.array([[np.cos(obj.ry),  0,  np.sin(obj.ry)],
                     [0,  1,  0],
                     [-(np.sin(obj.ry)), 0,  np.cos(obj.ry)]])

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R_yaw, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0]
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1]
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2]
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2,:] < 0.1):
        corners_2d = None
        return corners_2d, corners_3d.T


    pts_3d_extend = np.hstack((corners_3d.T, np.ones(((corners_3d.T).shape[0],1))))
    pts_2d = np.dot(pts_3d_extend, P.T) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    corners_2d = pts_2d[:,0:2]
    return corners_2d, corners_3d.T

def compute_orientation_3d(obj, P):
    # compute rotational matrix around yaw axis
    R_yaw = np.array([[np.cos(obj.ry),  0,  np.sin(obj.ry)],
                     [0,  1,  0],
                     [-(np.sin(obj.ry)), 0,  np.cos(obj.ry)]])
   
    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l],[0,0],[0,0]])
    
    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R_yaw, orientation_3d)
    orientation_3d[0,:] = orientation_3d[0,:] + obj.t[0]
    orientation_3d[1,:] = orientation_3d[1,:] + obj.t[1]
    orientation_3d[2,:] = orientation_3d[2,:] + obj.t[2]
    
    if np.any(orientation_3d[2,:]<0.1):
      orientation_2d = None
      return orientation_2d, orientation_3d.T
    
    pts_3d_extend = np.hstack((orientation_3d.T, np.ones(((orientation_3d.T).shape[0],1))))
    pts_2d = np.dot(pts_3d_extend, P.T) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    orientation_2d = pts_2d[:,0:2]
    return orientation_2d, orientation_3d.T
