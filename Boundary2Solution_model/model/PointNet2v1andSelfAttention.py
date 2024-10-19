import numpy as np
import torch
import torch.nn as nn

import math


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
                
        self.block1 = nn.Conv1d(in_channels=3,out_channels=64,kernel_size=1)
        self.block2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        self.block3 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=1)
        
        self.wq1 = nn.Linear(256,256)
        self.wk1 = nn.Linear(256,256)
        self.wv1 = nn.Linear(256,256)
        self.wq2 = nn.Linear(256,256)
        self.wk2 = nn.Linear(256,256)
        self.wv2 = nn.Linear(256,256)
        self.wq3 = nn.Linear(256,256)
        self.wk3 = nn.Linear(256,256)
        self.wv3 = nn.Linear(256,256)
        self.soft = nn.Softmax(dim=1)
        
        self.block4 = nn.Conv1d(in_channels=256,out_channels=1024,kernel_size=1)
    
    
        self.block5 = nn.Conv1d(in_channels=1024+128,out_channels=512,kernel_size=1)
        self.block6 = nn.Conv1d(in_channels=512,out_channels=256,kernel_size=1)
        self.block7 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1)
    
        self.bct1 = nn.BatchNorm1d(64)
        self.bct2 = nn.BatchNorm1d(128)
        self.bct3 = nn.BatchNorm1d(256)
        self.bct4 = nn.BatchNorm1d(1024)
        
        self.bct5 = nn.BatchNorm1d(512)
        self.bct6 = nn.BatchNorm1d(256)
        self.bct7 = nn.BatchNorm1d(128)
    
        self.block8 = nn.Conv1d(in_channels=128,out_channels=1,kernel_size=1)

    def forward(self, x):
        n = len(x)
        
        x = torch.relu(self.bct1(self.block1(x)))
        x = torch.relu(self.bct2(self.block2(x)))
        y = self.bct3(self.block3(x))
        
        y = y.view(y.size()[0:2])
        
        q = self.wq1(y)
        k = self.wk1(y)
        v = self.wv1(y)
        sa1 = self.soft(q@k.t()/np.sqrt(n))@v
        y = y + sa1
        
        q = self.wq2(y)
        k = self.wk2(y)
        v = self.wv2(y)
        sa2 = self.soft(q@k.t()/np.sqrt(n))@v
        y = y + sa2
        
        q = self.wq3(y)
        k = self.wk3(y)
        v = self.wv3(y)
        sa3 = self.soft(q@k.t()/np.sqrt(n))@v
        y = y + sa3
        
        y = y.view(y.size()[0],y.size()[1],1)
        
        y = torch.relu(self.bct4(self.block4(y)))
        
        
        y = torch.max(y, 0, keepdim=True)[0]

        y = y.repeat(n,1,1)

        x = torch.cat((x,y),1)
        

        x = torch.relu(self.bct5(self.block5(x)))
        x = torch.relu(self.bct6(self.block6(x)))
        x = torch.relu(self.bct7(self.block7(x)))
        
        x = self.block8(x)
        
        return x
    
