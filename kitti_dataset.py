import os
import torch
import torch.nn as nn

# from torch.utils.data import Dataset
import torch.utils.data as data

from PIL import Image
import torchvision.transforms as transforms

class KittiDataset(data.Dataset):

    class_names = ['Pedestrian', 'Car', 'Cyclist']
    class_numbers = {'Pedestrian': 0, 'Car': 1, 'Cyclist':2}
    
    def __init__(self, root_dir='D:\3D-Object-Detection-for-Autonomous-Driving\dataset\kitti',set='train', transform=transforms.Compose([
        transforms.Resize((375, 1242)),
        transforms.ToTensor(),
        # transforms.Resize((256,256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'training/image_2')
        self.labels_dir = os.path.join(root_dir, 'training/label_2')
        self.images = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].split('.')[0] + '.txt')

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)


        with open(label_path, 'r') as f:
            labels = f.readlines()

        return image, labels