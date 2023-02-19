import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import cv2
import math


object_list = {'Car':0, 'Van':1 , 'Truck':2 , 'Pedestrian':3 , 'Person_sitting':4 , 'Cyclist':5 , 'Tram':6 }
class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram' ]


bc={}
bc['minX'] = 0; bc['maxX'] = 80; bc['minY'] = -40; bc['maxY'] = 40

def interpret_kitti_label(bbox):
    w, h, l, y, z, x, yaw = bbox[8:15]
    y = -y
    yaw = (yaw + np.pi/2)

    return x, y, w, l, yaw

def get_target2(label_file):
    target = np.zeros([50, 7], dtype=np.float32)

    with open(label_file, 'r') as f:
        lines = f.readlines()

    index = 0

    for j in range(len(lines)):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()

        if obj_class in class_list:
            bbox = []
            bbox.append(object_list[obj_class])
            bbox.extend([float(e) for e in obj[1:]])

            x, y, w, l, yaw = interpret_kitti_label(bbox)

            location_x= x
            location_y= y

            if  (location_x>0) & (location_x<40) & (location_y>-40)  & (location_y<40) :
                target[index][2] = (y+ 40)/40            
                target[index][1] = x/40

                target[index][3]=float(l)/80
                target[index][4]=float(w)/40 

                target[index][5]=math.sin(float(yaw)) 
                target[index][6]=math.cos(float(yaw))


                for i in range(len(class_list)):
                    if obj_class == class_list[i]: 
                            target[index][0]=i
                index=index+1

    return target


def removePoints(PointCloud, BoundaryCond):
        
    # Remove points outside x,y range
    mask = np.where((PointCloud[:, 0] >= BoundaryCond['minX']) & (PointCloud[:, 0]<=BoundaryCond['maxX']) & (PointCloud[:, 1] >= BoundaryCond['minY']))
    PointCloud = PointCloud[mask]

    PointCloud[:,2] = PointCloud[:,2]+2
    return PointCloud

def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    # 1024 x 1024 x 3
    Height = 1024+1
    Width = 1024+1

    PointCloud = np.copy(PointCloud_)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization))
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization) + Width/2)
    
    indices = np.lexsort((-PointCloud[:,2],PointCloud[:,1],PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height,Width))

    _, indices = np.unique(PointCloud[:,0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    #some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:,0]), np.int_(PointCloud_frac[:,1])] = PointCloud_frac[:,2]


    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height,Width))
    densityMap = np.zeros((Height,Width))
    
    _, indices, counts = np.unique(PointCloud[:,0:2], axis = 0, return_index=True,return_counts = True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1)/np.log(64))
    
    intensityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = PointCloud_top[:,3]
    densityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = normalizedCounts

    RGB_Map = np.zeros((Height,Width,3))
    RGB_Map[:,:,0] = densityMap      # r_map
    RGB_Map[:,:,1] = heightMap       # g_map
    RGB_Map[:,:,2] = intensityMap    # b_map
    
    result = np.zeros((512,1024,3))
    result = RGB_Map[0:512,0:1024,:]
    return result





def get_target(label_file,calib_file):
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]

    cal = {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

    t=cal['Tr_velo2cam']
    target = np.zeros([50, 7], dtype=np.float32)
    
    with open(label_file,'r') as f:
        lines = f.readlines()

    index=0
    for j in range(len(lines)):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        #print(obj)


        if obj_class in class_list:
             
             t_lidar , _ = box3d_cam_to_velo(obj[8:], t)   # get target  3D object location x,y
             location_x = t_lidar[0][0]          
             location_y = t_lidar[0][1]

             if  (location_x>0) & (location_x<40) & (location_y>-40)  & (location_y<40) :
                  target[index][2] = t_lidar[0][0]/40    
                  target[index][1] = (t_lidar[0][1]+40)/80

                  obj_width  = obj[9].strip()
                  obj_length = obj[10].strip()
                  target[index][3]=float(obj_width)/80
                  target[index][4]=float(obj_length)/40     


                  obj_alpha = obj[3].strip()            # get target Observation angle of object, ranging [-pi..pi]
                  target[index][5]=math.sin(float(obj_alpha))
                  target[index][6]=math.cos(float(obj_alpha))
    
                  for i in range(len(class_list)):
                       if obj_class == class_list[i]:
                              target[index][0]=i
                  index=index+1

    return target


anchors = [[0.24,0.68], [0.27,0.33], [0.64,1.48], [0.70,1.82], [1.04,4.64]]

def bbox_iou(b1, b2, x1y1x2y2=True):
    #bounding box coordinates
    if not x1y1x2y2:
        b1_x1, b1_x2 = b1[:, 0] - b1[:, 2] / 2, b1[:, 0] + b1[:, 2] / 2
        b1_y1, b1_y2 = b1[:, 1] - b1[:, 3] / 2, b1[:, 1] + b1[:, 3] / 2
        b2_x1, b2_x2 = b2[:, 0] - b2[:, 2] / 2, b2[:, 0] + b2[:, 2] / 2
        b2_y1, b2_y2 = b2[:, 1] - b2[:, 3] / 2, b2[:, 1] + b2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    # get the corrdinates of the intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.max(torch.tensor(0), inter_x2 - inter_x1 + 1) * torch.max(torch.tensor(0), inter_y2 - inter_y1 + 1)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def box3d_cam_to_velo(box3d, Tr):

    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    T = np.zeros([4, 4], dtype=np.float32)
    T[:3, :] = Tr
    T[3, 3] = 1
    T_inv = np.linalg.inv(T)
    lidar_loc_ = np.dot(T_inv, cam)
    lidar_loc = lidar_loc_[:3]

    t_lidar =lidar_loc.reshape(1, 3)

    box = np.array([[-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                    [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                    [0, 0, 0, 0, h, h, h, h]])
    ang = -ry - np.pi / 2

    if ang >= np.pi:
        ang -= np.pi
    elif ang < -np.pi:
        ang = 2*np.pi + ang

    rz = ang

    rot = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                    [np.sin(rz), np.cos(rz), 0.0],
                    [0.0, 0.0, 1.0]])

    velo_box = np.dot(rot, box)
    velo_corner = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = velo_corner.T

    return t_lidar , box3d_corner.astype(np.float32)


def bbox_ious(b1, b2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(b1[0], b2[0])
        Mx = torch.max(b1[2], b2[2])
        my = torch.min(b1[1], b2[1])
        My = torch.max(b1[3], b2[3])
        w1 = b1[2] - b1[0]
        h1 = b1[3] - b1[1]
        w2 = b2[2] - b2[0]
        h2 = b2[3] - b2[1]
    else:
        mx = torch.min(b1[0]-b1[2]/2.0, b2[0]-b2[2]/2.0)
        Mx = torch.max(b1[0]+b1[2]/2.0, b2[0]+b2[2]/2.0)
        my = torch.min(b1[1]-b1[3]/2.0, b2[1]-b2[3]/2.0)
        My = torch.max(b1[1]+b1[3]/2.0, b2[1]+b2[3]/2.0)
        w1 = b1[2]
        h1 = b1[3]
        w2 = b2[2]
        h2 = b2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea






