import torch
import torch.nn as nn
import torch.utils.data as data
from complexyolo import COMPLEXYOLO

batch_size=4

# dataset
dataset=KittiDataset(root='D:/3D-Object-Detection-for-Autonomous-Driving/dataset/kitti',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True)

model = COMPLEXYOLO()
model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters())