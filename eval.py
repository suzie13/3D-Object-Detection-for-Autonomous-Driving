import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from yolo import YOLO
from kitti_dataset import KittiDataset
# from torchvision import transforms
import torchvision.transforms as transforms

# Load the pre-trained model
model = YOLO()
model.load_state_dict(torch.load('complexYOLO_10.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

kitti_dataset = KittiDataset(root_dir='D:\3D-Object-Detection-for-Autonomous-Driving\dataset\kitti',set='train', transform=transform)
data_loader = DataLoader(kitti_dataset, batch_size=32, shuffle=False, num_workers=4)

correct = 0
total = 0
with torch.no_grad():
    for data in data_loader:
        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d', (total, 100 * correct / total))