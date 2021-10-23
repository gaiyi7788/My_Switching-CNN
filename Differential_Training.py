import os
import torch
import torch.nn as nn

from models.model import Reg1,Reg2,Reg3
from data.dataset import CrowdDataset
from utils.input_split import input_split
from utils.plot_tensor import plot_tensor
from utils.min_loss_index import min_loss_index

def differential_training(R1,R2,R3,f):
    criterion=nn.L1Loss(size_average=False).to(device) #differencial_training用的是L1loss
    optimizer1 = torch.optim.SGD(R1.parameters(), lr=1e-6,
                                momentum=0.95)
    optimizer2 = torch.optim.SGD(R2.parameters(), lr=1e-6,
                                momentum=0.95)
    optimizer3 = torch.optim.SGD(R3.parameters(), lr=1e-6,
                                momentum=0.95)
    min_mae=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]

    for epoch in range(0,1000):
        R1.train()
        R2.train()
        R3.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(trainloader):
            img_list = input_split(img,3)
            gt_dmap_list = input_split(gt_dmap,3)
            # forward pass
            for (img_patch,gt_dmap_patch) in zip(img_list,gt_dmap_list):
                img_patch = img_patch.to(device)
                gt_dmap_patch = gt_dmap_patch.to(device)
                et_dmap_patch1 = R1(img_patch)
                et_dmap_patch2 = R2(img_patch)
                et_dmap_patch3 = R3(img_patch)
                
                loss1 = criterion(et_dmap_patch1.data.sum(),gt_dmap_patch.data.sum())
                loss2 = criterion(et_dmap_patch2.data.sum(),gt_dmap_patch.data.sum())
                loss3 = criterion(et_dmap_patch3.data.sum(),gt_dmap_patch.data.sum())
                loss, best_label = min_loss_index(loss1,loss2,loss3)
                # 使用.data.sum()以后得到的loss好像requires_grad属性不是True了，需要手动设置
                loss.requires_grad = True 
                epoch_loss+=loss.item()

                if best_label == 1:
                    optimizer1.zero_grad()
                    loss.backward()
                    optimizer1.step()   
                elif best_label == 2:
                    optimizer2.zero_grad()
                    loss.backward()
                    optimizer2.step()       
                elif best_label == 3:
                    optimizer3.zero_grad()
                    loss.backward()
                    optimizer3.step()                
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(trainloader))   

        R1.eval()
        R2.eval()
        R3.eval()
        mae=0
        for i,(img,gt_dmap) in enumerate(testloader):
            img_list = input_split(img,3)
            gt_dmap_list = input_split(gt_dmap,3)
            # forward propagation
            for (img_patch,gt_dmap_patch) in zip(img_list,gt_dmap_list):
                img_patch = img_patch.to(device)
                gt_dmap_patch = gt_dmap_patch.to(device)
                et_dmap_patch1 = R1(img_patch)
                et_dmap_patch2 = R2(img_patch)
                et_dmap_patch3 = R3(img_patch)
                loss1 = criterion(et_dmap_patch1.data.sum(),gt_dmap_patch.data.sum())
                loss2 = criterion(et_dmap_patch2.data.sum(),gt_dmap_patch.data.sum())
                loss3 = criterion(et_dmap_patch3.data.sum(),gt_dmap_patch.data.sum())
                loss, best_label = min_loss_index(loss1,loss2,loss3)
                mae+=loss.item() #按文章的意思是取所有最小的误差求和得到MAE
                del img_patch,gt_dmap_patch,et_dmap_patch1,et_dmap_patch2,et_dmap_patch3
        if mae/len(testloader)<min_mae: 
            min_mae=mae/len(testloader)
            min_epoch=epoch #那应该是保存总的MAE最小的一次的pt模型
            torch.save(R1.state_dict(),'checkpoints/R1_dif.pt')
            torch.save(R2.state_dict(),'checkpoints/R2_dif.pt')
            torch.save(R3.state_dict(),'checkpoints/R3_dif.pt')
        test_error_list.append(mae/len(testloader))    
        print("epoch:"+str(epoch)+" MAE:"+str(mae/len(testloader))+" min_MAE:"+str(min_mae)+" min_epoch:"+str(min_epoch))  
        f.write("epoch:"+str(epoch)+" MAE:"+str(mae/len(testloader))+" min_MAE:"+str(min_mae)+" min_epoch:"+str(min_epoch))   
        f.write("\n")        

if __name__=="__main__":
    torch.backends.cudnn.enabled=False #禁用非确定性算法
    device=torch.device("cuda")
    R1 = Reg1().to(device)
    R2 = Reg2().to(device)
    R3 = Reg3().to(device)
    R1.load_state_dict(torch.load('checkpoints/R1.pt'), strict=True)
    R2.load_state_dict(torch.load('checkpoints/R2.pt'), strict=True)
    R3.load_state_dict(torch.load('checkpoints/R3.pt'), strict=True)
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
    with open('checkpoints/Differential_Training.txt','w') as f:
        differential_training(R1,R2,R3,f)
