import numpy as np
import math
import cv2
import time
import torch

from model import *
import torch.utils.data as torch_data
from utils import*
from test_utils import *
from kitti_utils import *

from kitti_dataset import KITTI
import test_utils as mview

def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_size, RGB_Map=None):
    predictions = np.zeros([50, 7], dtype=np.float32)
    count = 0
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(im, re)
            predictions[count, :] = cls_pred, x/img_size, y/img_size, w/img_size, l/img_size, im, re
            count += 1

    predictions = yolo_labels_inv(predictions, boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = lidar_camera_bbox(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):

        str = "Pedestrian"
        if l[0] == 0:str="Car"
        elif l[0] == 1:str="Pedestrian"
        elif l[0] == 2: str="Cyclist"
        else:str = "DontCare"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = Object3d(line)
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

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = bev_labels(objects_new)    
        if not noObjectLabels:
            labels[:, 1:] = cam_lidar_bbox(labels[:, 1:], calib.V2C, calib.R0, calib.P) # convert rect cam to velo cord

        target = build_target(labels)
        draw_box_in_bev(RGB_Map, target)

    return objects_new

if __name__ == "__main__":

    fp = open("dataset/classes.names", "r")
    classes = fp.read().split("\n")[:-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = COMPLEXYOLO("config/complex_yolov3.cfg", img_size=608).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load("checkpoints/jkjas-74.pth"), strict = False)
    model.eval()
    
    dataset = KITTI('dataset', split='valid', mode='TEST', folder='training')
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    start_time = time.time()                        
    for index, (img_paths, bev_maps) in enumerate(data_loader):
        input_imgs = Variable(bev_maps.type(Tensor))
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression_rotated_bbox(detections, 0.5, 0.5) 
        
        end_time = time.time()
        print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
        start_time = end_time

        img_detections = []
        img_detections.extend(detections)

        bev_maps = torch.squeeze(bev_maps).numpy()

        RGB_Map = np.zeros((608, 608, 3))
        RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
        RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
        RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map
        
        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)
        
        for detections in img_detections:
            if detections is None:
                continue

            # Rescale boxes to original image
            detections = rescale_boxes(detections, 608, RGB_Map.shape[:2])
            for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                drawRotatedBox(RGB_Map, x, y, w, l, yaw, colors[int(cls_pred)])

        img2d = cv2.imread(img_paths[0])
        calib = Calibrate(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img2d.shape, 608)  
        
        img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)

        filename1 = f"image_{index}.jpg"
        filename2 = f"img_{index}.jpg"

        cv2.imwrite(filename1, RGB_Map)
        cv2.imwrite(filename2, img2d)
        