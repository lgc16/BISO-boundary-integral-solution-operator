import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as v
import scipy.special as scp
import random
from concurrent import futures
#import Exa_sol as ex
import time
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# 输出一维或二维的tensor或numpy元素到name.txt中
def Output2txt(x,path,name):
    if torch.is_tensor(x):
        x = x.detach().numpy()
    shape = x.shape
    if path==None:
        path = '.'
    f = open(path+'/'+name+'.txt','w')
    if len(shape)>1:
        for i in range(shape[0]):
            for j in range(shape[1]):
                #print(x[i,j],end=' ', file=f)
                f.write(str(x[i,j])+' ')
            #print('')
            f.write('\n')
    else:
        for i in range(shape[0]):
            #print(x[i], file=f)
            f.write(str(x[i])+'\n')
    f.close()
    return None

def Inputtxt(name,path=None,dtype=None):
    if path==None:
        path = '.'
    file = open(path+'/'+name+'.txt','r')
    file_data = file.readlines()
    for i in range(len(file_data)):
        file_data[i] = file_data[i].split()
        for j in range(len(file_data[i])):
            file_data[i][j] = float(file_data[i][j])
    if dtype == 'torch':
        file_data = torch.Tensor(file_data)
    if dtype == 'numpy':
        file_data = np.array(file_data)
    return file_data    

if __name__=="__main__":
    x = torch.rand(10,10)
    Output2txt(x,None,'test')
    

