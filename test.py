import os
import torch
import torch.nn as nn

import numpy as np
from scipy import io
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as CM

from models.model import Switch,Reg1,Reg2,Reg3
from data.dataset import CrowdDataset

device=torch.device("cuda")

def plot_dmap(et_dmap,gt_dmap):
    et_dmap = torch.squeeze(et_dmap).cpu().numpy()
    gt_dmap = torch.squeeze(gt_dmap).cpu().numpy()
    plt.subplot(121)
    plt.imshow(et_dmap,cmap=CM.jet)
    plt.title("et_dmap")
    plt.subplot(122)
    plt.imshow(gt_dmap,cmap=CM.jet)
    plt.title("gt_dmap")
    plt.show()
    
    

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

test_dataset_root = "data/ShanghaiTech/part_A/test_data"
test_img_root = os.path.join(test_dataset_root,'images')
test_gt_dmap_root = os.path.join(test_dataset_root,'gt_dmaps')
testset = CrowdDataset(test_img_root,test_gt_dmap_root,4)
testloader=torch.utils.data.DataLoader(testset,batch_size=1,shuffle=True)
model_dict_path = "checkpoints/best.pt"


switch = Switch().to(device)
switch.load_state_dict(torch.load("checkpoints/switch_Coupled.pt"), strict=False)
R1 = Reg1().to(device)
R1.load_state_dict(torch.load("checkpoints/R1_Coupled.pt"), strict=True)
R2 = Reg2().to(device)
R2.load_state_dict(torch.load("checkpoints/R2_Coupled.pt"), strict=True)
R3 = Reg3().to(device)
R3.load_state_dict(torch.load("checkpoints/R3_Coupled.pt"), strict=True)

# model.cuda()
switch.eval()
R1.eval()
R2.eval()
R3.eval()
mae = 0.0
mse = 0.0

with torch.no_grad():
    for i,(img,gt_dmap) in enumerate(testloader):
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        pred = switch(img)
        label = pred.argmax(dim = 1).cpu().numpy()[0]
        if label == 0:
            R = R1
        elif label == 1:
            R = R2
        elif label == 2:
            R = R3
        else:
            print("--------------Error!!!!!----------------")
        et_dmap = R(img)
        # et_dmap = torch.clamp(et_dmap,0)
        # zero = torch.zeros_like(et_dmap)
        # et_damp = torch.where(et_dmap<0,zero,et_dmap)
        # et_dmap = et_dmap/torch.max(et_dmap)
        plot_dmap(et_dmap,gt_dmap)
        print("finish")