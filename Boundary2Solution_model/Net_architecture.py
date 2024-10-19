import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as v

import time
import matplotlib.pyplot as plt
import math


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        
        n = 1000
        
        self.block1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=[1,3])
        self.block2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=[1,1])
        
        self.block3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=[1,1])
        self.block4 = nn.Conv2d(in_channels=128,out_channels=1024,kernel_size=[1,1])
        
        self.pool = nn.MaxPool2d(kernel_size=[n,1])
        
        self.block5 = nn.Conv2d(in_channels=1088,out_channels=512,kernel_size=[1,1])
        self.block6 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=[1,1])
        self.block7 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=[1,1])
        
        self.block8 = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=[1,1])

    def forward(self, x):
        n = len(x[0][0])
        x = torch.relu(self.block1(x))
        x = torch.relu(self.block2(x))
        
        y = torch.relu(self.block3(x))
        y = torch.relu(self.block4(y))
        y = self.pool(y)
        y = y.reshape(1,-1)
        y0 = y
        for i in range(n-1):
            y0 = torch.cat((y0,y),0)
        x = ((x[0]).reshape(-1,n)).permute(1,0)
        x = torch.cat((x,y0),1)
        
        x = x.permute(1,0)
        x = x.reshape([1,1088,n,1])

        x = torch.relu(self.block5(x))
        x = torch.relu(self.block6(x))
        x = torch.relu(self.block7(x))
        
        x = self.block8(x)
        
        return x
