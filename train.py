import os
import torch
import torch.nn as nn

from models.model import Switching_CNN
from data.dataset import CrowdDataset

def train():
    torch.backends.cudnn.enabled=False #禁用非确定性算法
    device=torch.device("cuda")
    model = Switching_CNN(True,0).to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6,
                                momentum=0.95)
    # root = os.getcwd()
    train_dataset_root = "dataset/ShanghaiTech/part_A/train_data"
    train_img_root = os.path.join(train_dataset_root,'images')
    train_gt_dmap_root = os.path.join(train_dataset_root,'gt_dmaps')
    trainset = CrowdDataset(train_img_root,train_gt_dmap_root,4)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True)
    
    test_dataset_root = "dataset/ShanghaiTech/part_A/test_data"
    test_img_root = os.path.join(test_dataset_root,'images')
    test_gt_dmap_root = os.path.join(test_dataset_root,'gt_dmaps')
    testset = CrowdDataset(test_img_root,test_gt_dmap_root,4)
    testloader=torch.utils.data.DataLoader(testset,batch_size=1,shuffle=True)
    
    #training phase
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    min_mae=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    
    for epoch in range(0,2000):
        model.train()
        epoch_loss=0
        for i,(img,gt_map) in enumerate(trainloader):
            img = img.to(device)
            gt_map = gt_map.to(device)
            # forward pass
            et_map = model(img)
            loss = criterion(et_map,gt_map)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(trainloader))
                 
        model.eval()
        mae=0
        for i,(img,gt_dmap) in enumerate(testloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap
        if mae/len(testloader)<min_mae:
            min_mae=mae/len(testloader)
            min_epoch=epoch
            torch.save(model.state_dict(),'./checkpoints/best.pt') 
        test_error_list.append(mae/len(testloader))    
        print("epoch:"+str(epoch)+" error:"+str(mae/len(testloader))+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))     
    
if __name__=="__main__":
    train()