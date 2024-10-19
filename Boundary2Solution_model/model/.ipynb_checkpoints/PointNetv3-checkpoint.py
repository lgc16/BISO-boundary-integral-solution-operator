## 针对最朴素的PointNetv1的一个改进
#  改进一、Conv2d改成了Conv1d，这是为了用Batchnorm1d
#  改进二、读了GitHub上程序才发现每一层都加了batchnorm1d，遂加上，原文中在图注里讲了，没看到

#  待改进、两个Tnet即transform变换没有加入进去


import numpy as np
import torch
import torch.nn as nn

import math


class PointNet1(nn.Module):
    def __init__(self,input_num1,input_num2,out_num):
        super(PointNet, self).__init__()
        
        self.num1 = input_num1
        self.block11 = nn.Conv1d(in_channels=input_num1,out_channels=64,kernel_size=1)
        self.block21 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1)
    
        self.block31 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        self.block41 = nn.Conv1d(in_channels=128,out_channels=512,kernel_size=1)
    
        self.block12 = nn.Conv1d(in_channels=input_num2,out_channels=64,kernel_size=1)
        self.block22 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1)
    
        self.block32 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        self.block42 = nn.Conv1d(in_channels=128,out_channels=512,kernel_size=1)
    
    
        self.block5 = nn.Conv1d(in_channels=1152,out_channels=512,kernel_size=1)
        self.block6 = nn.Conv1d(in_channels=512,out_channels=256,kernel_size=1)
        self.block7 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1)
    
        self.bct11 = nn.BatchNorm1d(64)
        self.bct21 = nn.BatchNorm1d(64)
        self.bct31 = nn.BatchNorm1d(128)
        self.bct41 = nn.BatchNorm1d(512)
        
        self.bct12 = nn.BatchNorm1d(64)
        self.bct22 = nn.BatchNorm1d(64)
        self.bct32 = nn.BatchNorm1d(128)
        self.bct42 = nn.BatchNorm1d(512)
        
        self.bct5 = nn.BatchNorm1d(512)
        self.bct6 = nn.BatchNorm1d(256)
        self.bct7 = nn.BatchNorm1d(128)
    
        self.block8 = nn.Conv1d(in_channels=128,out_channels=out_num,kernel_size=1)

    def forward(self, x):
        x1,x2 = x.split(self.num1,1)
        n = len(x)
        
        x1 = torch.relu(self.bct11(self.block11(x1)))
        x1 = torch.relu(self.bct21(self.block21(x1)))
        
        y1 = torch.relu(self.bct31(self.block31(x1)))
        y1 = torch.relu(self.bct41(self.block41(y1)))
        y1 = torch.max(y1, 0, keepdim=True)[0]

        
        
        x2 = torch.relu(self.bct12(self.block12(x2)))
        x2 = torch.relu(self.bct22(self.block22(x2)))
        
        y2 = torch.relu(self.bct32(self.block32(x2)))
        y2 = torch.relu(self.bct42(self.block42(y2)))
        y2 = torch.max(y2, 0, keepdim=True)[0]
        
        
        
        x = torch.cat((x1,x2),1)
        y = torch.cat((y1,y2),1)
        
        
        y = y.repeat(n,1,1)

        x = torch.cat((x,y),1)
        

        x = torch.relu(self.bct5(self.block5(x)))
        x = torch.relu(self.bct6(self.block6(x)))
        x = torch.relu(self.bct7(self.block7(x)))
        
        x = self.block8(x)
        
        return x

    
    
## 针对最朴素的PointNetv1的一个改进
#  改进一、Conv2d改成了Conv1d，这是为了用Batchnorm1d
#  改进二、读了GitHub上程序才发现每一层都加了batchnorm1d，遂加上，原文中在图注里讲了，没看到

#  待改进、两个Tnet即transform变换没有加入进去


import numpy as np
import torch
import torch.nn as nn

import math


class PointNet(nn.Module):
    def __init__(self,input_num1,input_num2,out_num):
        super(PointNet, self).__init__()
        
        self.num1 = input_num1
        self.block11 = nn.Conv1d(in_channels=input_num1,out_channels=64,kernel_size=1)
        self.block21 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1)
    
        self.block31 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        self.block41 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=1)
    
        self.block12 = nn.Conv1d(in_channels=input_num2,out_channels=64,kernel_size=1)
        self.block22 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1)
    
        self.block32 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        self.block42 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=1)
        
        self.block1 = nn.Conv1d(in_channels=input_num1+input_num2,out_channels=64,kernel_size=1)
        self.block2 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1)
    
        self.block3 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        self.block4 = nn.Conv1d(in_channels=128,out_channels=512,kernel_size=1)
    
    
        self.block5 = nn.Conv1d(in_channels=1216,out_channels=512,kernel_size=1)
        self.block6 = nn.Conv1d(in_channels=512,out_channels=256,kernel_size=1)
        self.block7 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1)
    
        self.bct1 = nn.BatchNorm1d(64)
        self.bct2 = nn.BatchNorm1d(64)
        self.bct3 = nn.BatchNorm1d(128)
        self.bct4 = nn.BatchNorm1d(512)
        
        self.bct11 = nn.BatchNorm1d(64)
        self.bct21 = nn.BatchNorm1d(64)
        self.bct31 = nn.BatchNorm1d(128)
        self.bct41 = nn.BatchNorm1d(256)
        
        self.bct12 = nn.BatchNorm1d(64)
        self.bct22 = nn.BatchNorm1d(64)
        self.bct32 = nn.BatchNorm1d(128)
        self.bct42 = nn.BatchNorm1d(256)
        
        self.bct5 = nn.BatchNorm1d(512)
        self.bct6 = nn.BatchNorm1d(256)
        self.bct7 = nn.BatchNorm1d(128)
    
        self.block8 = nn.Conv1d(in_channels=128,out_channels=out_num,kernel_size=1)

    def forward(self, x):
        x1,x2 = x.split(self.num1,1)
        n = len(x)
        
        x = torch.relu(self.bct1(self.block1(x)))
        x = torch.relu(self.bct2(self.block2(x)))

        y = torch.relu(self.bct3(self.block3(x)))
        y = torch.relu(self.bct4(self.block4(y)))
        y = torch.max(y, 0, keepdim=True)[0]

        
        x1 = torch.relu(self.bct11(self.block11(x1)))
        x1 = torch.relu(self.bct21(self.block21(x1)))
        
        y1 = torch.relu(self.bct31(self.block31(x1)))
        y1 = torch.relu(self.bct41(self.block41(y1)))
        y1 = torch.max(y1, 0, keepdim=True)[0]
        
        
        x2 = torch.relu(self.bct12(self.block12(x2)))
        x2 = torch.relu(self.bct22(self.block22(x2)))
        
        y2 = torch.relu(self.bct32(self.block32(x2)))
        y2 = torch.relu(self.bct42(self.block42(y2)))
        y2 = torch.max(y2, 0, keepdim=True)[0]
        
        
        
        x = torch.cat((x,torch.cat((x1,x2),1)),1)
        y = torch.cat((y,torch.cat((y1,y2),1)),1)
        
        
        y = y.repeat(n,1,1)

        x = torch.cat((x,y),1)
        

        x = torch.relu(self.bct5(self.block5(x)))
        x = torch.relu(self.bct6(self.block6(x)))
        x = torch.relu(self.bct7(self.block7(x)))
        
        x = self.block8(x)
        
        return x
