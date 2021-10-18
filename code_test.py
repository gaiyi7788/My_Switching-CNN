import torch
import torch.nn as nn
import os
from data.dataset import CrowdDataset
from models.model import Switching_CNN

if __name__=="__main__": 
    train_dataset_root = "data/ShanghaiTech/part_A/train_data"
    train_img_root = os.path.join(train_dataset_root,'images')
    train_gt_dmap_root = os.path.join(train_dataset_root,'gt_dmaps')
    trainset = CrowdDataset(train_img_root,train_gt_dmap_root,4)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True)
    
    model = Switching_CNN()
    feature = torch.nn.Sequential(*list(model.children())[:])
    print(feature)
    
    for i,(img,gt_dmap) in enumerate(trainloader):
        output = model(img)
        print(output.shape)