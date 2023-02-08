import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from yolo import COMPLEXYOLO
from kitti_dataset import KittiDataset
from region_loss import RegionLoss


batch_size=12

# dataset
dataset=KittiDataset(root='D:/3D-Object-Detection-for-Autonomous-Driving/dataset/kitti',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True)

model = COMPLEXYOLO()
model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
region_loss = RegionLoss(num_classes=8)



for epoch in range(200):

    for batch_idx, (rgb_map, target) in enumerate(data_loader):          
          optimizer.zero_grad()
         #  print(rgb_map.shape)
          rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
         #  print(rgb_map.shape)
       #    print(output.shape)
          output = model(rgb_map.float().cuda())
         #  print(output.shape)
         #  print(target.shape)
         #  print(".....................")

          loss = region_loss(output,target)

          loss.backward()
          optimizer.step()

    torch.save(model, "COMPLEXYOLO_epoch"+str(epoch))
    # torch.save(model, "CY_epoch"+str(epoch))
#    torch.save(model.state_dict(), 'epoch_%d.pth' % (epoch + 1))
    

print('Finished Training')
