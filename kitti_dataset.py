from __future__ import division
import os
import os.path
import torch
import numpy as np
import glob
import torch.utils.data as torch_data
from kitti_utils import *
import random

from utils import *

import torch
import torch.nn.functional as F


class KittiDataset(torch_data.Dataset):

    def __init__(self, root_dir, split='train', folder='testing'):
        self.split = split

        is_test = self.split == 'test'
        self.folder_name = os.path.join('data\KITTI\object', folder)
        self.lidar_path = os.path.join(self.folder_name, "velodyne")

        self.image_path = os.path.join(self.folder_name, "image_2")
        self.calib_path = os.path.join(self.folder_name, "calib")
        self.label_path = os.path.join(self.folder_name, "label_2")
        print(self.label_path)


        if not is_test:
            split_dir = os.path.join('data', 'KITTI', 'ImageSets', split+'.txt')
            self.img_list = [x.strip() for x in open(split_dir).readlines()]
        else:
            self.files = sorted(glob.glob("%s/*.bin" % self.lidar_path))
            self.img_list = [os.path.split(x)[1].split(".")[0].strip() for x in self.files]
            print(self.img_list[0])

        self.num_samples = len(self.img_list)
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
        objects = [Class_3d(line) for line in lines]
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
        self.preprocess_training_data()

    def preprocess_training_data(self):

        for idx in range(0, self.num_samples):
        # for idx in range(0, 300):
            file = int(self.img_list[idx])
            # print(int(self.img_list[224]))
            print(file)
            objects = self.get_label(file)
            calib = self.get_calib(file)
            labels, noObjectLabels = bev_labels(objects)
            if not noObjectLabels:
                labels[:, 1:] = cam_lidar_bbox(labels[:, 1:], calib.V2C, calib.R0, calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in object_list.values():
                    x_range = [bc["minX"], bc["maxX"]]
                    y_range = [bc["minY"], bc["maxY"]]
                    z_range = [bc["minZ"], bc["maxZ"]]

                    if (x_range[0] <= (labels[i, 1:4])[0] <= x_range[1]) and (y_range[0] <= (labels[i, 1:4])[1] <= y_range[1]) and \
                            (z_range[0] <= (labels[i, 1:4])[2] <= z_range[1]):
                        valid_list.append(labels[i,0])

            if len(valid_list):
                self.file_list.append(file)

    def __getitem__(self, index):
        
        file = int(self.file_list[index])

        if self.mode in ['TRAIN', 'EVAL']:
            lidarData = self.get_lidar(file)    
            objects = self.get_label(file)   
            calib = self.get_calib(file)

            labels, noObjectLabels = bev_labels(objects)
    
            if not noObjectLabels:
                labels[:, 1:] = cam_lidar_bbox(labels[:, 1:], calib.V2C, calib.R0, calib.P) 

            b = removePoints(lidarData, bc)
            rgb_map = makeBVFeature(b, DISCRETIZATION, bc)
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
            b = removePoints(lidarData, bc)
            rgb_map = makeBVFeature(b, DISCRETIZATION, bc)
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
