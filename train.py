import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from yolo import YOLO
from kitti_dataset import KittiDataset
# from torchvision import transforms



# Load the Kitti dataset
kitti_dataset = KittiDataset()
data_loader = DataLoader(kitti_dataset, batch_size=8, shuffle=True)

# Initialize the YOLO model
model = YOLO()
# model.cuda()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        #check here 
        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(data_loader)))
    torch.save(model.state_dict(), 'complexYOLO_%d.pth' % (epoch + 1))

print('Finished Training')
