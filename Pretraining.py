import os
import torch
import torch.nn as nn

from models.model import Reg1,Reg2,Reg3
from data.dataset import CrowdDataset
from utils.input_split import input_split
from utils.plot_tensor import plot_tensor

def pretraining(R, label, f, trainloader, testloader):
    optimizer = torch.optim.SGD(R.parameters(), lr=1e-6,
                                momentum=0.95)
    min_mae=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    
    for epoch in range(0,100):
        R.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(trainloader):
            img_list = input_split(img,3)
            gt_dmap_list = input_split(gt_dmap,3)
            # forward pass
            for (img_patch,gt_dmap_patch) in zip(img_list,gt_dmap_list):
                # plot_tensor(img_patch,gt_dmap_patch) #可视化显示输入样例用
                img_patch = img_patch.to(device)
                gt_dmap_patch = gt_dmap_patch.to(device)
                et_dmap_patch = R(img_patch)
                loss = criterion(et_dmap_patch,gt_dmap_patch)
                epoch_loss+=loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()            
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(trainloader))    
        
        R.eval()
        mae=0
        for i,(img,gt_dmap) in enumerate(testloader):
            img_list = input_split(img,3)
            gt_dmap_list = input_split(gt_dmap,3)
            # forward propagation
            for (img_patch,gt_dmap_patch) in zip(img_list,gt_dmap_list):
                img_patch = img_patch.to(device)
                gt_dmap_patch = gt_dmap_patch.to(device)
                et_dmap_patch=R(img_patch)
                mae+=abs(et_dmap_patch.data.sum()-gt_dmap_patch.data.sum()).item()
                del img_patch,gt_dmap_patch,et_dmap_patch
        if mae/len(testloader)<min_mae:
            min_mae=mae/len(testloader)
            min_epoch=epoch
            torch.save(R.state_dict(),'./checkpoints/'+label+'.pt')
        test_error_list.append(mae/len(testloader))    
        print(label+"---"+"epoch:"+str(epoch)+" error:"+str(mae/len(testloader))+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))  
        f.write(label+"---"+"epoch:"+str(epoch)+" error:"+str(mae/len(testloader))+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))   
        f.write("\n")

if __name__=="__main__":
    torch.backends.cudnn.enabled=False #禁用非确定性算法
    device=torch.device("cuda")
    R1 = Reg1().to(device)
    R2 = Reg2().to(device)
    R3 = Reg3().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    # root = os.getcwd()
    train_dataset_root = "data/ShanghaiTech/part_A/train_data"
    train_img_root = os.path.join(train_dataset_root,'images')
    train_gt_dmap_root = os.path.join(train_dataset_root,'gt_dmaps')
    trainset = CrowdDataset(train_img_root,train_gt_dmap_root,4,3)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True)
    
    test_dataset_root = "data/ShanghaiTech/part_A/test_data"
    test_img_root = os.path.join(test_dataset_root,'images')
    test_gt_dmap_root = os.path.join(test_dataset_root,'gt_dmaps')
    testset = CrowdDataset(test_img_root,test_gt_dmap_root,4,3)
    testloader=torch.utils.data.DataLoader(testset,batch_size=1,shuffle=True)
    
    #training phase
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
        
    with open('checkpoints/R1_Pretraining.txt','w') as f1:
        pretraining(R1,'R1',f1,trainloader,testloader)
    with open('checkpoints/R2_Pretraining.txt','w') as f2:
        pretraining(R2,'R2',f2,trainloader,testloader)
    with open('checkpoints/R3_Pretraining.txt','w') as f3:
        pretraining(R3,'R3',f3,trainloader,testloader)
