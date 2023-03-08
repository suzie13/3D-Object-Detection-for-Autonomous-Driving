from __future__ import division

from utils import *
from model import KITTI
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp = open("data/classes.names", "r")
    class_names = fp.read().split("\n")[:-1]


    model = COMPLEXYOLO("config/complex_yolov3.cfg", img_size=608).to(device)
    model.apply(weights_init_normal)

    dataset = KITTI('data',split='train',mode='TRAIN',folder='training')

    dataloader = DataLoader(dataset,4,shuffle=True,num_workers=8,pin_memory=True,collate_fn=dataset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters())

    metrics = ["grid_size","loss","x","y","w","h","im","re","conf","cls","cls_acc","recall50","recall75","precision","conf_obj","conf_noobj"]

    max_mAP = 0.0
    for epoch in range(0, 300):
        model.train()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % 2:
                optimizer.step()
                optimizer.zero_grad()


            model.seen += imgs.size(0)

            torch.save(model.state_dict(), f"checkpoints/kkkkajjhjfy-%d.pth" % (epoch))


