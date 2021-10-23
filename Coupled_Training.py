import os
import torch
import torch.nn as nn
from random import sample

from models.model import Switch,Reg1,Reg2,Reg3
from data.dataset import CrowdDataset
from utils.input_split import input_split
from utils.plot_tensor import plot_tensor
from utils.min_loss_index import min_loss_index

device=torch.device("cuda")

def Gen_SW_labels(img_patch,gt_dmap_patch,R1,R2,R3):
    criterion=nn.L1Loss(size_average=False).to(device)
    et_dmap_patch1 = R1(img_patch)
    et_dmap_patch2 = R2(img_patch)
    et_dmap_patch3 = R3(img_patch)
    loss1 = criterion(et_dmap_patch1.data.sum(),gt_dmap_patch.data.sum())
    loss2 = criterion(et_dmap_patch2.data.sum(),gt_dmap_patch.data.sum())
    loss3 = criterion(et_dmap_patch3.data.sum(),gt_dmap_patch.data.sum())
    _, best_label = min_loss_index(loss1,loss2,loss3)
    return best_label

def random_balance(samples1, samples2, samples3):
    # print(len(samples1)," ", len(samples2)," ", len(samples3))
    num = min(len(samples1), len(samples2), len(samples3))
    samples1 = sample(samples1,num)
    samples2 = sample(samples2,num)
    samples3 = sample(samples3,num)
    # print(len(samples1)," ", len(samples2)," ", len(samples3))
    return samples1,samples2,samples3

def Gen_samples(R1,R2,R3,trainloader):
    samples1 = []
    samples2 = []
    samples3 = []
    for i,(img,gt_dmap) in enumerate(trainloader):
        img_list = input_split(img,3)
        gt_dmap_list = input_split(gt_dmap,3)
        # forward pass
        for (img_patch,gt_dmap_patch) in zip(img_list,gt_dmap_list):
            img_patch = img_patch.to(device)
            gt_dmap_patch = gt_dmap_patch.to(device)  
            best_label = Gen_SW_labels(img_patch,gt_dmap_patch,R1,R2,R3) 
            sample = (img_patch,best_label,gt_dmap_patch)    
            if best_label == 1:
                samples1.append(sample)
            elif best_label == 2:
                samples2.append(sample) 
            elif best_label == 3:
                samples3.append(sample)  
    return random_balance(samples1, samples2, samples3)

def Switch_Training(switch,samples):
    criterion=nn.CrossEntropyLoss().to(device)
    epoch_loss = 0
    optimizer = torch.optim.SGD(switch.parameters(), lr=1e-6,
                                momentum=0.95)
    for samples_k in samples:
        for (img_patch,best_label,_) in samples_k:
            pred = switch(img_patch)
            #best_label-1转为[0,1,2]
            label = torch.unsqueeze(torch.tensor(best_label-1).to(device),dim = 0)
            loss = criterion(pred,label) 
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

def SD_Training(switch,R1,R2,R3,samples): # samples contains： sample = (img_patch,best_label,gt_dmap_patch)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer1 = torch.optim.SGD(R1.parameters(), lr=1e-6,
                                momentum=0.95)
    optimizer2 = torch.optim.SGD(R2.parameters(), lr=1e-6,
                                momentum=0.95)
    optimizer3 = torch.optim.SGD(R3.parameters(), lr=1e-6,
                                momentum=0.95)
    epoch_loss=0
    switch.train()
    R1.train()
    R2.train()
    R3.train()
    for samples_k in samples:
        for (img_patch,_,gt_dmap_patch) in samples_k:
            pred = switch(img_patch)
            label = pred.argmax(dim = 1).cpu().numpy()[0]
            if label == 0:
                R = R1
                optimizer = optimizer1
            elif label == 1:
                R = R2
                optimizer = optimizer2
            elif label == 2:
                R = R3
                optimizer = optimizer3
            else:
                print("--------------Error!!!!!----------------")
            et_dmap_patch = R(img_patch)
            loss = criterion(et_dmap_patch,gt_dmap_patch)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

def Coupled_eval(switch,R1,R2,R3,testloader,epoch,min_epoch,min_MAE):
    criterion=nn.L1Loss(size_average=False).to(device)
    switch.eval()
    R1.eval()
    R2.eval()
    R3.eval()
    MAE=0
    for i,(img,gt_dmap) in enumerate(testloader):
        img_list = input_split(img,3)
        gt_dmap_list = input_split(gt_dmap,3)
        # forward propagation
        for (img_patch,gt_dmap_patch) in zip(img_list,gt_dmap_list):
            img_patch = img_patch.to(device)
            gt_dmap_patch = gt_dmap_patch.to(device)
            pred = switch(img_patch)
            label = pred.argmax(dim = 1).cpu().numpy()[0]
            if label == 0:
                R = R1
            elif label == 1:
                R = R2
            elif label == 2:
                R = R3
            else:
                print("--------------Error!!!!!----------------")
            et_dmap_patch = R(img_patch)
            loss = criterion(et_dmap_patch.data.sum(),gt_dmap_patch.data.sum())
            MAE+=loss.item() #按文章的意思是取所有最小的误差求和得到MAE
            del img_patch,gt_dmap_patch,et_dmap_patch

    if MAE/len(testloader)<min_MAE: 
        min_MAE=MAE/len(testloader)
        min_epoch=epoch #那应该是保存总的MAE最小的一次的pt模型
        torch.save(R1.state_dict(),'checkpoints/switch_Coupled.pt')
        torch.save(R1.state_dict(),'checkpoints/R1_Coupled.pt')
        torch.save(R2.state_dict(),'checkpoints/R2_Coupled.pt')
        torch.save(R3.state_dict(),'checkpoints/R3_Coupled.pt')   

    return MAE,min_epoch,min_MAE          

def Coupled_Training(trainloader,testloader):
    switch = Switch().to(device)
    R1 = Reg1().to(device)
    R2 = Reg2().to(device)
    R3 = Reg3().to(device)
    R1.load_state_dict(torch.load('checkpoints/R1_dif.pt'), strict=True)
    R2.load_state_dict(torch.load('checkpoints/R2_dif.pt'), strict=True)
    R3.load_state_dict(torch.load('checkpoints/R3_dif.pt'), strict=True)
    min_MAE = 10000
    min_epoch = 0
    with open('checkpoints/Coupled_Training.txt','w') as f:
        for epoch in range(0,1000):
            # 样本生成
            samples1,samples2,samples3 = Gen_samples(R1,R2,R3,trainloader)
            samples = (samples1,samples2,samples3)
            R1.train()
            R2.train()
            R3.train()
            # 训练 Switch
            Switch_Training(switch,samples)       
            # 利用 switch 的预测的标签来训练 Differencial 
            SD_Training(switch,R1,R2,R3,samples)
            # 根据 MAE 评价模型
            with torch.no_grad():
                MAE,min_epoch,min_MAE = Coupled_eval(switch,R1,R2,R3,testloader,epoch,min_epoch,min_MAE)
                print("epoch:"+str(epoch)+" MAE:"+str(MAE/len(testloader))+" min_MAE:"+str(min_MAE)+" min_epoch:"+str(min_epoch))  
                f.write("epoch:"+str(epoch)+" MAE:"+str(MAE/len(testloader))+" min_MAE:"+str(min_MAE)+" min_epoch:"+str(min_epoch))   
                f.write("\n")  


if __name__=="__main__":
    torch.backends.cudnn.enabled=False #禁用非确定性算法
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
    Coupled_Training(trainloader,testloader)