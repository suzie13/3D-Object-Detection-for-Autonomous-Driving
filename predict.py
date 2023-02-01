import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from yolo import YOLO


def convert_to_bounding_box(output):
    output = output.detach().numpy()
    # Get bounding boxes
    bounding_boxes = output[:, :4]
    # Normalize the bounding boxes
    bounding_boxes = bounding_boxes * (img_width, img_height, img_width, img_height)
    return bounding_boxes

def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Load pre-trained model
model = YOLO()
model.load_state_dict(torch.load('complexYOLO_10.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load image
image_path = 'D:\3D-Object-Detection-for-Autonomous-Driving\dataset\kitti\testing\image_2\000010.png'
img = Image.open(image_path)
img = transform(img)
img = img.unsqueeze(0) # add a batch dimension

# Predict the bounding boxes
outputs = model(img)

bounding_boxes = []
for output in outputs:
    # Convert output to bounding box
    # TODO
    bounding_box = convert_to_bounding_box(output)
    bounding_boxes.append(bounding_box)

# Show the image with the bounding boxes
img = img.squeeze(0) # remove the batch dimension
img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img) # denormalize the image
show_image(img)
for bounding_box in bounding_boxes:
    x1, y1, x2, y2 = bounding_box
    #cv2.rectangle is also an option
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'))
plt.show()