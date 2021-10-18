import os
import torch
import torch.nn as nn

import numpy as np
from scipy import io
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as CM

from model import MCNN
from dataset import CrowdDataset


def plot_dmap(et_dmap,gt_dmap):
    et_dmap = torch.squeeze(et_dmap).numpy()
    gt_dmap = torch.squeeze(gt_dmap).numpy()
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

test_dataset_root = "dataset/ShanghaiTech/part_A/test_data"
test_img_root = os.path.join(test_dataset_root,'images')
test_gt_dmap_root = os.path.join(test_dataset_root,'gt_dmaps')
testset = CrowdDataset(test_img_root,test_gt_dmap_root,4)
testloader=torch.utils.data.DataLoader(testset,batch_size=1,shuffle=True)
model_dict_path = "checkpoints/best.pt"


model = MCNN()
model.load_state_dict(torch.load(model_dict_path), strict=True)

# model.cuda()
model.eval()
mae = 0.0
mse = 0.0

with torch.no_grad():
    for i,(img,gt_dmap) in enumerate(testloader):
        et_dmap = model(img)
        plot_dmap(et_dmap,gt_dmap)