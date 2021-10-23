from scipy.integrate._ivp.dop853_coefficients import D
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
import torch.nn.functional as F
import numpy as np
import os
from data.dataset import CrowdDataset

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, bn_act=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if kernel_size==3:
            padding = 1 
        elif kernel_size==5:
            padding = 2
        elif kernel_size==7:
            padding = 3
        elif kernel_size==9:
            padding = 4
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = not bn_act)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)
    
class Switch(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG16 = vgg16_bn(num_classes = 3)
    def forward(self, x):
        x = self.VGG16(x)
        return x
        # x = F.softmax(self.VGG16(x),dim=1)
        # x = x.detach().numpy()[0]
        # print(x)
        # label = np.where(x==np.max(x))
        # print(label)
        # return label[0][0]   

class Reg1(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.net = nn.Sequential(
            Conv(3,16,9),
            self.maxpool,
            Conv(16,32,7),
            self.maxpool,
            Conv(32,16,7),
            Conv(16,8,7),
            Conv(8,1,1)
        )
    def forward(self, x):
        x = self.net(x)
        return x

class Reg2(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.net = nn.Sequential(
            Conv(3,20,7),
            self.maxpool,
            Conv(20,40,5),
            self.maxpool,
            Conv(40,20,5),
            Conv(20,10,5),
            Conv(10,1,1)
        )
    def forward(self, x):
        x = self.net(x)
        return x

class Reg3(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.net = nn.Sequential(
            Conv(3,16,9),
            self.maxpool,
            Conv(16,32,7),
            self.maxpool,
            Conv(32,16,7),
            Conv(16,8,7),
            Conv(8,1,1)
        )
    def forward(self, x):
        x = self.net(x)
        return x
    
class Switching_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.switch = Switch()
        self.R1 = Reg1()
        self.R2 = Reg2()
        self.R3 = Reg3()
        
    def forward(self, x):
        label = self.switch(x)
        if label == 0:
            x = self.R1(x)
            print("R1")
        elif label == 1:
            x = self.R2(x)
            print("R2")
        elif label == 2:
            x = self.R3(x)
            print("R3")
        return x
