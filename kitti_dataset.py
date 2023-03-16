from __future__ import division
import os
import numpy as np
import random
from utils import *

import torch
import torch.nn.functional as F

import glob
import torch.utils.data as torch_data
from kitti_utils import *

class KittiDataset(torch_data.Dataset):

    def __init__(self, root_dir, split='train', folder='testing'):
        self.split = split

        is_test = self.split == 'test'
        self.folder_name = os.path.join('dataset\KITTI\object', folder)
        self.lidar_path = os.path.join(self.folder_name, "velodyne")

        self.image_path = os.path.join(self.folder_name, "image_2")
        self.calib_path = os.path.join(self.folder_name, "calib")
        self.label_path = os.path.join(self.folder_name, "label_2")
        print(self.label_path)


        if not is_test:
            split_dir = os.path.join('dataset', 'KITTI', 'ImageSets', split+'.txt')
            self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        else:
            self.files = sorted(glob.glob("%s/*.bin" % self.lidar_path))
            self.image_idx_list = [os.path.split(x)[1].split(".")[0].strip() for x in self.files]
            print(self.image_idx_list[0])

        # self.num_samples = self.image_idx_list.__len__()
        self.num_samples = len(self.image_idx_list)
        # self.num_samples = 300
        print(self.num_samples)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibrate(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_path, '%06d.txt' % idx)
        print(label_file)
        lines = [line.rstrip() for line in open(label_file)]
        objects = [Object3d(line) for line in lines]
        return objects


class KITTI(KittiDataset):

    def __init__(self, root_dir, split='train', mode ='TRAIN', folder=None):
        super().__init__(root_dir=root_dir, split=split, folder=folder)

        self.split = split
        self.min_size = 608 - 3 * 32
        self.max_size = 608 + 3 * 32
        self.batch_count = 0

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        self.file_list = []
        self.preprocess_yolo_training_data()

    def preprocess_yolo_training_data(self):

        for idx in range(0, self.num_samples):
        # for idx in range(0, 300):
            file = int(self.image_idx_list[idx])
            # print(int(self.image_idx_list[224]))
            print(file)
            objects = self.get_label(file)
            calib = self.get_calib(file)
            labels, noObjectLabels = bev_labels(objects)
            if not noObjectLabels:
                labels[:, 1:] = cam_lidar_bbox(labels[:, 1:], calib.V2C, calib.R0, calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in object_list.values():
                    x_range = [boundary["minX"], boundary["maxX"]]
                    y_range = [boundary["minY"], boundary["maxY"]]
                    z_range = [boundary["minZ"], boundary["maxZ"]]

                    if (x_range[0] <= (labels[i, 1:4])[0] <= x_range[1]) and (y_range[0] <= (labels[i, 1:4])[1] <= y_range[1]) and \
                            (z_range[0] <= (labels[i, 1:4])[2] <= z_range[1]):
                        valid_list.append(labels[i,0])

            if len(valid_list):
                self.file_list.append(file)

    def __getitem__(self, index):
        
        file = int(self.file_list[index])
        # print(len(file))

        if self.mode in ['TRAIN', 'EVAL']:
            lidarData = self.get_lidar(file)    
            objects = self.get_label(file)   
            calib = self.get_calib(file)

            labels, noObjectLabels = bev_labels(objects)
    
            if not noObjectLabels:
                labels[:, 1:] = cam_lidar_bbox(labels[:, 1:], calib.V2C, calib.R0, calib.P)


            b = removePoints(lidarData, boundary)
            rgb_map = makeBVFeature(b, DISCRETIZATION, boundary)
            target = build_target(labels)

            img_file = os.path.join(self.image_path, '%06d.png' % file)

            ntargets = 0
            for i, t in enumerate(target):
                if t.sum(0):
                    ntargets += 1            
            targets = torch.zeros((ntargets, 8))
            for i, t in enumerate(target):
                if t.sum(0):
                    targets[i, 1:] = torch.from_numpy(t)
            
            img = torch.from_numpy(rgb_map).type(torch.FloatTensor)

            return img_file, img, targets

        else:
            lidarData = self.get_lidar(file)
            b = removePoints(lidarData, boundary)
            rgb_map = makeBVFeature(b, DISCRETIZATION, boundary)
            img_file = os.path.join(self.image_path, '%06d.png' % file)
            return img_file, rgb_map

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.stack([F.interpolate(img.unsqueeze(0), size=608, mode="nearest").squeeze(0) for img in imgs])

        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.file_list)
