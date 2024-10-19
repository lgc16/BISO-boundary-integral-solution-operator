# 这一篇是从pointnet像pointnet++的一个过渡工作
# 基于PointNetv2进行改进
# 很多的Pointnet++的内容没放进来，主要就是加一个局部特征的提取encode和一个通过插值上采样decode

import numpy as np
import torch
import torch.nn as nn

import math


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
                
        self.block1 = nn.Conv1d(in_channels=3,out_channels=64,kernel_size=1)
        self.block2 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1)
    
        self.block3 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        self.block4 = nn.Conv1d(in_channels=128,out_channels=1024,kernel_size=1)
    
    
        self.block5 = nn.Conv1d(in_channels=1088,out_channels=512,kernel_size=1)
        self.block6 = nn.Conv1d(in_channels=512,out_channels=256,kernel_size=1)
        self.block7 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1)
    
        self.bct1 = nn.BatchNorm1d(64)
        self.bct2 = nn.BatchNorm1d(64)
        self.bct3 = nn.BatchNorm1d(128)
        self.bct4 = nn.BatchNorm1d(1024)
        
        self.bct5 = nn.BatchNorm1d(512)
        self.bct6 = nn.BatchNorm1d(256)
        self.bct7 = nn.BatchNorm1d(128)
    
        self.block8 = nn.Conv1d(in_channels=128,out_channels=1,kernel_size=1)

    def forward(self, x):
        n = len(x)
        
        x = torch.relu(self.bct1(self.block1(x)))
        x = torch.relu(self.bct2(self.block2(x)))
        
        y = torch.relu(self.bct3(self.block3(x)))
        y = torch.relu(self.bct4(self.block4(y)))
        
        
        y = torch.max(y, 0, keepdim=True)[0]

        y = y.repeat(n,1,1)

        x = torch.cat((x,y),1)
        

        x = torch.relu(self.bct5(self.block5(x)))
        x = torch.relu(self.bct6(self.block6(x)))
        x = torch.relu(self.bct7(self.block7(x)))
        
        x = self.block8(x)
        
        return x
