import numpy as np
import torch
import torch.nn as nn

import math


# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py    
'''
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
'''


## 自己写了一段错误的FPS后终于理解了上面的FPS的意思，
## “最远”中的“距离”不是待选点到已选所有点的距离的平方，而是待选的点到已选的点的距离的最小值作为点到点集的距离，按照前一种会造成每次选择的点在两个点之间来回飘
def farthest_point_sample(x,npoints):
    """
    一共有B组点集，要从每一组点集里面选出npoint个点
    x : pointcloud, [B, N, 2]
    
    """
    #device = x.device
    N, d = x.shape
    far_index = torch.randint(0,N,(1,))
    distance = torch.ones(N)*1e10
    centroids = torch.zeros(npoints,2)
    #centroid_index = []
    for i in range(npoints):
        centroids[i,:] = x[far_index,:]
        dist = torch.sum((x-centroids[i,:])**2,-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        far_index = torch.max(distance,-1)[1]
        
    return centroids   
        
    
    
    
    