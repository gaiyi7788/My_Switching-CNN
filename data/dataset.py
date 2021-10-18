import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root,gt_downsample=1):
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample
        self.img_names = os.listdir(img_root)
        self.img_nums = len(self.img_names)
        
    def __len__(self):
        return self.img_nums
        
    def __getitem__(self,index):
        # 加一句判断index是否越界
        assert index <= len(self)
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_root,img_name)
        gt_dmap_name = img_name.replace('.jpg','.npy')
        gt_dmap_path = os.path.join(self.gt_dmap_root,gt_dmap_name)
        # img = plt.imread(img_path)
        img = cv2.imread(img_path)
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)
        
        ds_rows=int(img.shape[0]//self.gt_downsample)
        ds_cols=int(img.shape[1]//self.gt_downsample)
        img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
        img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
        gt_dmap = np.load(gt_dmap_path)
        gt_dmap = cv2.resize(gt_dmap,(ds_cols,ds_rows)) #因为gt_map最后是要和特征图大小相等，只用降采样即可
        # 目前不是很理解为什么要乘 self.gt_downsample
        gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample
        img=torch.tensor(img,dtype=torch.float)
        gt_dmap=torch.tensor(gt_dmap,dtype=torch.float)   
        return img, gt_dmap
        
if __name__=="__main__":
    dataset_root = "data/ShanghaiTech/part_A/train_data"
    img_root = os.path.join(dataset_root,'images')
    gt_dmap_root = os.path.join(dataset_root,'gt_dmaps')
    dataset = CrowdDataset(img_root,gt_dmap_root,16)
    for i,(img,gt_dmap) in enumerate(dataset):
        print(img.shape,gt_dmap.shape)