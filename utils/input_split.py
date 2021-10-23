import os
import torch
import torch.nn as nn

def input_split(input_tensor,split_scale):
    [N,C,H,W] = input_tensor.shape
    split_list = []
    h = int(H/split_scale)
    w = int(W/split_scale)
    split_list.append(input_tensor[:,:,:h,:w])
    split_list.append(input_tensor[:,:,h:2*h,:w])
    split_list.append(input_tensor[:,:,2*h:,:w])
    split_list.append(input_tensor[:,:,:h,w:2*w])
    split_list.append(input_tensor[:,:,h:2*h,w:2*w])
    split_list.append(input_tensor[:,:,2*h:,w:2*w])
    split_list.append(input_tensor[:,:,:h,2*w:])
    split_list.append(input_tensor[:,:,h:2*h,2*w:])
    split_list.append(input_tensor[:,:,2*h:,2*w:])
    return split_list