import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

def plot_tensor(img,gt_dmap):
    img = torch.squeeze(img, dim = 0)
    img = img.numpy().transpose((1,2,0))/255
    plt.subplot(121)
    plt.imshow(img)
    gt_dmap = torch.squeeze(gt_dmap, dim = 0)
    gt_dmap = gt_dmap.numpy().transpose((1,2,0))/255
    plt.subplot(122)
    plt.imshow(gt_dmap)
    
    plt.savefig('test/testshow.png', bbox_inches='tight')
    plt.show()