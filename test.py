import numpy as np
import math
import os
import argparse
import cv2
import time
import torch

# import utils.utils as utils
from model import *
import torch.utils.data as torch_data
from utils import*

from kitti_utils import *

from kitti_dataset import KITTI
from test_utils import *

def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_size, RGB_Map=None):
    cnt = 0
    predictions = np.zeros([50, 7], dtype=np.float32)
    for detections in img_detections:
        if detections is None:
            continue
        for x, y, w, l, imagin, real, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(imagin, real)
            predictions[cnt, :] = cls_pred, x/img_size, y/img_size, w/img_size, l/img_size, imagin, real
            cnt += 1


    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):

        str = "Pedestrian"
        if l[0] == 0:str="Car"
        elif l[0] == 1:str="Pedestrian"
        elif l[0] == 2: str="Cyclist"
        else:str = "DontCare"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = Class_3d(line)
        obj.t = l[1:4]
        obj.h,obj.w,obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))
    
        _, corners_3d = compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]


    return objects_new

if __name__ == "__main__":

    fp = open("data/classes.names", "r")
    classes = fp.read().split("\n")[:-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = COMPLEXYOLO("config/complex_yolov3.cfg", img_size=608).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load("checkpoints\qwsasaf-100.pth"), strict = False)
    model.eval()
    
    # dataset = KITTI('data', split='valid', mode='TEST', folder='training', data_aug=False)
    dataset = KITTI('data', split='valid', mode='TEST', folder='training')
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    start_time = time.time()                        
    for index, (img_paths, bev_maps) in enumerate(data_loader):
        
        # bev image
        input_imgs = Variable(bev_maps.type(Tensor))


        bev_maps = torch.squeeze(bev_maps).numpy()

        RGB_Map = np.zeros((608, 608, 3))
        RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
        RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
        RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map
        
        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)
        

        img2d = cv2.imread(img_paths[0])
        calib = Calibrate(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        
        filename1 = f"image_{index}.jpg"
        filename2 = f"img_{index}.jpg"

        cv2.imwrite(filename1, RGB_Map)
        cv2.imwrite(filename2, img2d)
        