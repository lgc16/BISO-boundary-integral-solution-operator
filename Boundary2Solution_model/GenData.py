import numpy as np
import torch
import torch.nn as nn
import scipy.special as scp
from concurrent import futures
#import Exa_sol as ex
import time
import math
from tqdm import tqdm


def gen_bd_int_points(vertex_x,N):    
    sample_tri = len(vertex_x)
    vertex_num = len(vertex_x[0])
    rot = torch.Tensor([[0,-1],[1,0]])  #顺时针旋转90度
    
    num_each =  torch.zeros(vertex_num)
    
    for i in range(vertex_num-1):
        num_each[i] = int(round(N/vertex_num))
    num_each[vertex_num-1] = int(N-round(N/vertex_num)*(vertex_num-1))

    # 生成多边形上相应的积分点
    x = torch.zeros(sample_tri,N,2)
    normal = torch.zeros(sample_tri,N,2)            #h_i 为第i个点和第i+1个点之间的外法向量
    h = torch.zeros(sample_tri,N,1)                 #h_i 为第i个点和第i+1个点之间的距离
    c = torch.zeros(sample_tri,N,2)
    
    # 生成逐段的外法向量和计算权重用到的量 
    #for j in tqdm(range(sample_tri)):
    for j in (range(sample_tri)):
        k = 0
        for i in range(vertex_num):
            line = (torch.linspace(0,1-1/num_each[i],int(num_each[i]))).reshape(-1,1)
            x[j,k:k+int(num_each[i]),:] = line*(vertex_x[j,(i+1)%vertex_num,:]-vertex_x[j,i,:])+vertex_x[j,i,:]
            normal[j,k:k+int(num_each[i]),:] = (vertex_x[j,(i+1)%vertex_num,:]-vertex_x[j,i,:])@rot
            normal[j,k:k+int(num_each[i]),:] = normal[j,k:k+int(num_each[i]),:]/((normal[j,k:k+int(num_each[i]),0]**2+normal[j,k:k+int(num_each[i]),1]**2).sqrt().reshape(-1,1))
            h[j,k:k+int(num_each[i])] = (vertex_x[j,(i+1)%vertex_num,:]-vertex_x[j,i,:]).norm()/num_each[i]
            k = k + int(num_each[i])
        a = normal[j,:,:]*h[j,:]
        b = torch.cat((a[N-1,:].reshape(1,2),a[1:N,:]),0)
        c[j,:,:] = (a+b)/2
        
    return x, c, normal, h, num_each


def gen_polygon(vertex_num,N,sample_tri,a1,a2,b1,b2):
    ### 生成多边形和边界上的积分节点
    ### 但是生成积分节点打包成了gen_bd_int_points
    
    # 生成多边形
    theta = torch.rand(sample_tri,vertex_num)*a1+a2
    r = torch.rand(sample_tri,vertex_num)*b1+b2

    theta_sum = theta.sum(axis=1).reshape(-1,1)
    for i in range(vertex_num-1):
        theta[:,i+1] = theta[:,i+1] + theta[:,i]
        
    theta = theta/theta_sum*np.pi*2

    x1 = torch.cos(theta)*r
    x2 = torch.sin(theta)*r
    
    vertex_x = torch.zeros(sample_tri,vertex_num,2)     
    
    vertex_x[:,:,0] = x1
    vertex_x[:,:,1] = x2
    x, c, normal, h, num_each = gen_bd_int_points(vertex_x,N)
    return vertex_x , x, c, normal, h, num_each


def gen_rectangle(vertex_num,N,sample_tri,a1,a2,b1,b2):
    ### 生成多边形和边界上的积分节点
    ### 但是生成积分节点打包成了gen_bd_int_points
    
    # 生成多边形
    theta = torch.rand(sample_tri,vertex_num)*a1+a2
    theta[:,2:] = theta[:,0:2] 
    r = b2

    theta_sum = theta.sum(axis=1).reshape(-1,1)
    
    for i in range(vertex_num-1):
        theta[:,i+1] = theta[:,i+1] + theta[:,i]
        
    theta = theta/theta_sum*np.pi*2
    theta = theta - theta[:, 0].reshape(-1,1)
    

    rot = (np.pi-theta[:, 1].reshape(-1,1))/2 #+1/4*np.pi#np.random.rand(1)*b1*np.pi
    x1 = torch.cos(theta + rot)*r
    x2 = torch.sin(theta + rot)*r
    vertex_x = torch.zeros(sample_tri,vertex_num,2)     
    
    vertex_x[:,:,0] = x1
    vertex_x[:,:,1] = x2
    x, c, normal, h, num_each = gen_bd_int_points(vertex_x,N)
    return vertex_x , x, c, normal, h, num_each




def G_bd(vertex_x, x, c, num_each,in_or_out):
    if in_or_out == 'in':
        const = -1/2
    elif in_or_out == 'out':
        const = 1/2
    else:
        print('未知类型')
    
    N = len(x[0])
    vertex_num = len(vertex_x[0])
    ver_num = torch.zeros(vertex_num)
    for i in range(len(num_each)-1):
        ver_num[i+1] = ver_num[i] + num_each[i]
    sample_tri = len(vertex_x)
    G2 = torch.zeros(sample_tri,N,N)
    
    #for p in tqdm(range(sample_tri)):
    for p in (range(sample_tri)):
        for i in range(N):
            if (not (i in ver_num)):
                r = x[p,i,:]-x[p,:,:]
            
                p_n = (c[p,:,:]*r).sum(axis=1)
                d = (np.sqrt(r[:,0]**2+r[:,1]**2))
                G2[p,i,:] = - ( -1/(2*np.pi)*(p_n)/(d*d))
                G2[p,i,i] = const
    return G2   


def G_hel_bd(vertex_x, x, c, num_each,in_or_out,k):
    
    if in_or_out == 'in':
        const = 1/2
    elif in_or_out == 'out':
        const = -1/2
    else:
        print('未知类型')
    
    N = len(x[0])
    vertex_num = len(vertex_x[0])
    ver_num = torch.zeros(vertex_num)
    for i in range(len(num_each)-1):
        ver_num[i+1] = ver_num[i] + num_each[i]
    sample_tri = len(vertex_x)
    G2_r = torch.zeros(sample_tri,N,N)
    G2_i = torch.zeros(sample_tri,N,N)
    
    #for p in tqdm(range(sample_tri)):
    for p in (range(sample_tri)):
        for i in range(N):
            if (not (i in ver_num)):
                r = x[p,i,:]-x[p,:,:]
            
                p_n = (c[p,:,:]*r).sum(axis=1)
                d = (np.sqrt(r[:,0]**2+r[:,1]**2))
                
                G2_r[p,i,:] = -( scp.hankel1(1,k*d).imag/4*(p_n)/d*k )
                G2_i[p,i,:] = -(-scp.hankel1(1,k*d).real/4*(p_n)/d*k )
                G2_r[p,i,i] , G2_i[p,i,i] = -const , 0  
    return G2_r, G2_i   




def gen_points_in(vertex_x,num_in):
    # 生成 vertex_x 所含多边形的内部点 x_in
    sample_tri = len(vertex_x)
    x_in = torch.zeros(sample_tri,num_in,2)
    for p in range(sample_tri):
        x_in[p,:,:] = gen_point_in(vertex_x[p,:,:],num_in)
         
    return x_in  
    
    
def G_in(x_in, x, c):
    # 生成 内部点 相对于 边界点的积分矩阵 G2
    N = len(x[0])
    sample_tri = len(x)
    num_in = len(x_in[0])
    G2 = torch.zeros(sample_tri,num_in,N)
    for p in (range(sample_tri)):     
        for i in range(num_in):
            r = x[p,:,:]-x_in[p,i,:]
            d = (r**2).sum(axis=1).sqrt()
            G2[p,i,:] = -(-1/(2*np.pi)*((r*c[p,:,:]).sum(axis=1))/(d*d))
    return G2 


def G_Hel_in(x_in,x,c,k):
    
    # 生成 内部点 相对于 边界点的Helmholtz方程的积分矩阵 G2_r,G2_i
    N = len(x[0])
    sample_tri = len(x)
    num_in = len(x_in[0])
    G2_r_in = torch.zeros(sample_tri,num_in,N)
    G2_i_in = torch.zeros(sample_tri,num_in,N)
    for p in tqdm(range(sample_tri)):     
        for i in range(num_in):
            r = x[p,:,:]-x_in[p,i,:]
            d = (r**2).sum(axis=1).sqrt()
            
            G2_r_in[p,i,:] = scp.hankel1(1,k*d).imag/4*((r*c[p,:,:]).sum(axis=1))/d*k 
            G2_i_in[p,i,:] = -scp.hankel1(1,k*d).real/4*((r*c[p,:,:]).sum(axis=1))/d*k 
    
    return G2_r_in, G2_i_in


def bd_sol_process(num_each,u):
    # 边界解的处理
    # 因为顶点处的积分权重全部设为0 ，所以边界条件相应的值也要设为0
    
    k = 0
    for i in range(len(num_each)):
        u[:,k] = 0
        k = k + int(num_each[i])
    return u
    

            
### 生成多边形 vertex 内部的 num_in 个点
def gen_point_in(vertex,num_in):
    xmax , xmin = (vertex[:,0]).max() , (vertex[:,0]).min()
    ymax , ymin = (vertex[:,1]).max() , (vertex[:,1]).min()
    xy0 = torch.Tensor([xmax-xmin,ymax-ymin])
    xy1 = torch.Tensor([xmin,ymin])
    x = torch.rand(num_in,2)*xy0+xy1
    for i in range(num_in):
        while not inpoly(x[i,:],vertex):
            x[i,:] = torch.rand(1,2)*xy0+xy1
    return x            


### 这里是判断点 x 是否在多边形 vertex 中，
### 用了射线法，向x正半轴方向拉一条射线判断交点个数
### 可能不足够准确，因为一些边缘情况没有判断，比如点在边上，点在顶点上，但是目前看起来够用了
def inpoly(x,vertex):
    l = len(vertex)
    xmax , xmin = (vertex[:,0]).max() , (vertex[:,0]).min()
    x0 = x + torch.Tensor([1+xmax-x[1],0])        ## x x0是一条正方向的射线                    
    k = 0
    for i in range(l):
        x1 , x2 = vertex[i-1,:] , vertex[i,:]
        if ((x1[1]<x[1]) != (x2[1]<x[1])) and ((x1[1]>x[1]) != (x2[1]>x[1])):
            
            if x1[1]==x2[1]:
                if x1[1] == x[1]:
                    k = k + 1
            else:
                x01 = (x1[0]-x2[0])/(x1[1]-x2[1])*(x[1]-x1[1])+x1[0]
                if x01>=x[0]:
                    k = k + 1
    if k%2==0:
        return False
    else:
        return True

            

def exact_u(x,vertex_x,sol_type):
    # 生成 x 位置的精确解 
    # x 相应的坐标，vertex_x 多边形坐标， sol_type 解的类型
    xshape = x.shape
    vershape = vertex_x.shape
    if len(xshape)<3:
        x = x.reshape(1,xshape[0],xshape[1])
        vertex_x = vertex_x.reshape(1,vershape[0],vershape[1])
        
    l,l0 = x.shape[0],x.shape[1]
    u = torch.zeros(l,l0)
    a , b , c , d = 1 , 1 , 1 , 1
    for i in range(l):
        if sol_type[i]==0 or sol_type[i]==3:
            if sol_type[i]==3:
                a , b , c , d = 2 , 1 , 1 , 1
            else:
                a , b , c , d = 1 , 1 , 1 , 1
            u[i,:] = (a*x[i,:,0]*x[i,:,1]+b*x[i,:,0]+c*x[i,:,1]+d)
        elif (sol_type[i]==1):
            COG = vertex_x[i].mean(axis=0)
            x1 = x[i,:,0]-COG[0]
            x2 = x[i,:,1]-COG[1]
            u[i,:] = (a*x1*x2+b*x1+c*x2+d)
        elif (sol_type[i]==2):
            COG = vertex_x[i].mean(axis=0)
            x1 = x[i,:,0]-COG[0]-1.5
            x2 = x[i,:,1]-COG[1]-1.5
            u[i,:] = torch.log((x1**2+x2**2).sqrt())
        elif (sol_type[i]==10):
            x1, x2 = x[i,:,0], x[i,:,1]
            u[i,:] = -(torch.exp(torch.sin(x1)-2*torch.cos(x2)+1)+x1**2-2*x2)
        elif (sol_type[i]==11):
            x1, x2 = x[i,:,0], x[i,:,1]
            u[i,:] = -(x1**2+x2**2)
            
        else :
            return 0
    u = u.reshape([l,l0,1])
    if len(xshape)<3:
        return u[0]
    else:
        return u

    

def exact_un(x,vertex_x,normal,sol_type):
    ### 求边界上的法向导数
    ### 理论上应该把法向错一位相加求平均，否则在顶点处会出问题，但是因积分矩阵本身就舍掉了顶点，所以先不管了！！！！
    ### 还有问题没解决 ###
    ### 还有问题没解决 ###
    ### 还有问题没解决 ###
    ### 还有问题没解决 ###
    xshape = x.shape
    vershape = vertex_x.shape
    if len(xshape)<3:
        x = x.reshape(1,xshape[0],xshape[1])
        vertex_x = vertex_x.reshape(1,vershape[0],vershape[1])
    l,l0 = xshape[0],xshape[1]
    un = torch.zeros(l,l0)
    normal1 = np.zeros(normal.shape)
    normal1[:,1:,:] = normal[:,:l0-1,:]
    normal1[:,0,:] = normal[:,l0-1,:]
    normal1 = (normal+normal1)/2
    a , b , c , d = 1 , 1 , 1 , 1
    for i in range(l):
        if sol_type[i]==0 or sol_type[i]==3:
            if sol_type[i]==3:
                a , b , c , d = 2 , 1 , 1 , 1
            else:
                a , b , c , d = 1 , 1 , 1 , 1
            un[i,:] = ((a*x[i,:,1]+b)*normal1[i,:,0]+(a*x[i,:,0]+c)*normal1[i,:,1])
        elif sol_type[i]==1:
            COG = vertex_x[i].mean(axis=0)
            x1 = x[i,:,0]-COG[0]
            x2 = x[i,:,1]-COG[1]
            un[i,:] = ((a*x2+b)*normal1[i,:,0]+(a*x1+c)*normal1[i,:,1])
        elif (sol_type[i]==2):
            COG = vertex_x[i].mean(axis=0)
            x1 = x[i,:,0]-COG[0]-1.5
            x2 = x[i,:,1]-COG[1]-1.5
            un[i,:] = (x1*normal1[i,:,0]+x2*normal1[i,:,1])/(x1**2+x2**2)
    
    un = un.reshape([l,l0,1])
        
    if len(xshape)<3:
        return un[0]
    else:
        return un

def exact_u_hel(x,vertex_x,sol_type,k):
    # 生成 x 位置的精确解 
    # x 相应的坐标，vertex_x 多边形坐标， sol_type 解的类型
    xshape = x.shape
    vershape = vertex_x.shape
    if len(xshape)<3:
        x = x.reshape(1,xshape[0],xshape[1])
        vertex_x = vertex_x.reshape(1,vershape[0],vershape[1])
    a , b , c , d = 1 , 1 , 1 , 1
    if (sol_type==0) :
        l = len(vertex_x)
        l0 = len(x[0])
        
        theta0 = np.pi/5
        k1,k2 = k*np.cos(theta0) , k*np.sin(theta0) 
                
        u_r = (torch.cos(k1*x[:,:,0]+k2*x[:,:,1])).reshape([l,l0,1])
        u_i = (torch.sin(k1*x[:,:,0]+k2*x[:,:,1])).reshape([l,l0,1])
    elif (sol_type == 1):
        theta0 = np.pi/5
        k1,k2 = k*np.cos(theta0) , k*np.sin(theta0)
        COG = vertex_x.mean(axis=1)
        l = len(COG)
        l0 = len(x[0])
        u_r = torch.zeros(l,l0)
        u_i = torch.zeros(l,l0)
        for i in range(len(COG)):
            x1 = x[i,:,0]-COG[i,0]
            x2 = x[i,:,1]-COG[i,1]
            u_r[i,:] = torch.cos(k1*x1+k2*x2)
            u_i[i,:] = torch.sin(k1*x1+k2*x2)
        u_r = u_r.reshape([l,l0,1])
        u_i = u_i.reshape([l,l0,1])
    elif (sol_type==2):
        COG = vertex_x.mean(axis=1)
        l = len(COG)
        l0 = len(x[0])
        u_r = torch.zeros(l,l0)
        u_i = torch.zeros(l,l0)
        for i in range(len(COG)):
            x1 = x[i,:,0]-COG[i,0]-1.5
            x2 = x[i,:,1]-COG[i,1]-1.5
            scp.hankel1(1,k*d).imag
            u_r[i,:] = scp.hankel1(1,k*(x1**2+x2**2).sqrt()).real 
            u_i[i,:] = scp.hankel1(1,k*(x1**2+x2**2).sqrt()).imag 
        u_r = u_r.reshape([l,l0,1])
        u_i = u_i.reshape([l,l0,1])
    
    else :
        return 0
    
    if len(xshape)<3:
        return u[0]
    else:
        return u_r,u_i

def G_single_bd(vertex_x, x, h, num_each):
    kp2 = torch.Tensor([1.825748064736159e+00,-1.325748064736159e+00])
    N = len(x[0])
    one = torch.ones(N)
    one[1:len(kp2)+1] = one[1:len(kp2)+1]+kp2
    
    vertex_num = len(vertex_x[0])
    ver_num = torch.zeros(vertex_num)
    for i in range(len(num_each)-1):
        ver_num[i+1] = ver_num[i] + num_each[i]
    sample_tri = len(vertex_x)
    G2 = torch.zeros(sample_tri,N,N)
    index = np.array([i%N for i in range(N)])

    for p in (range(sample_tri)):
        for i in range(N):
            if (not (i in ver_num))and(not ((i+1)%N in ver_num))and(not (i-1 in ver_num)):
                r = x[p,i,:]-x[p,:,:]            
                d = (np.sqrt(r[:,0]**2+r[:,1]**2))
                G2[p,i,:] = -1/(2*np.pi)*torch.log(d)*(h[p].reshape(1,-1)[0])*one[abs(index-i)]
                G2[p,i,i] = 0
    return G2   


    
def G_single_in(x_in, x, h):
    # 生成 内部点 相对于 边界点的积分矩阵 G2
    N = len(x[0])
    sample_tri = len(x)
    num_in = len(x_in[0])
    G1 = torch.zeros(sample_tri,num_in,N)
    for p in (range(sample_tri)):     
        for i in range(num_in):
            r = x[p,:,:]-x_in[p,i,:]
            d = (r**2).sum(axis=1).sqrt()
            G1[p,i,:] = -1/(2*np.pi)*(torch.log(d)*(h[p].reshape(1,-1)[0]))
    return G1 

def become_zero(u,num_each,num):
    s = u.shape
    if len(s)==2:
        k = 0
        for i in range(len(num_each)):
            for j in range(num+1):
                u[:,k-j] = 0
                u[:,k+j] = 0            
            k = k + int(num_each[i])
    if len(s)==3:
        k = 0
        for i in range(len(num_each)):
            for j in range(num+1):
                u[:,k-j,:] = 0
                u[:,k+j,:] = 0            
            k = k + int(num_each[i])
    return u

def G_bd_normal(x1,normal1,x2,c2,num_each2): #这里应该可以用c2优化一下，看看怎么做
    sample_tri = len(x1)
    N = len(x1[0])
    G_bd_n = torch.zeros(sample_tri,N,len(x2[0]))
    ver_num = torch.zeros(len(num_each2))
    for i in range(len(num_each2)-1):
        ver_num[i+1] = ver_num[i] + num_each2[i]
    r1 = torch.zeros(len(x2[0]),2)
    for p in range(sample_tri):
        for i in range(N):
            if not (i in ver_num):
                r = x1[p,i,:] - x2[p]
                d = r.norm(dim=1)
                r1[:,0] = (r[:,0]**2-r[:,1]**2)*normal1[p,i,0] + 2*normal1[p,i,1]*(r[:,0]*r[:,1])
                r1[:,1] = (r[:,1]**2-r[:,0]**2)*normal1[p,i,1] + 2*normal1[p,i,0]*(r[:,1]*r[:,0])
                G_bd_n[p,i,:] = -(r1*c2[p]).sum(axis=1)/d**4/2/np.pi
    return G_bd_n

def G_bd_px(x,normal,h,num_each,in_or_out):
    #计算partial G(x,y)/partial x
    if in_or_out == 'in':
        c = -1/2
    else:
        c = 1/2
    sample_tri = len(x)
    N = len(x[0])
    G_px = torch.zeros(sample_tri,N,N)
    ver_num = torch.zeros(len(num_each))
    for i in range(len(num_each)-1):
        ver_num[i+1] = ver_num[i] + num_each[i]
    for p in range(sample_tri):
        for i in range(N):
            if not (i in ver_num):
                r = x[p,i,:]-x[p]
                d = r.norm(dim=1)
                G_px[p,i,:] = (r*normal[p,i,:]).sum(axis=1)/d**2/2/np.pi*(h[p].reshape(1,-1)[0])
                G_px[p,i,i] = c  ### 内部这里应该要变成-1/2
    return G_px


def gen_interface_polygon(domaintype,sample_tri,vertex_num1,N1,a11,a21,b11,b21,vertex_num2,N2,a12,a22,b12,b22):
    vertex_x1, x1, c1, normal1, h1, num_each1 = gen_polygon(vertex_num1,N1,sample_tri,a11,a21,b11,b21)  
    if domaintype==0:
        vertex_x2, x2, c2, normal2, h2, num_each2 = gen_polygon(vertex_num2,N2,sample_tri,a12,a22,b12,b22)  
    elif domaintype==1:
        diam = 1.5
        vertex0 = torch.Tensor([[-diam,-diam],[diam,-diam],[diam,diam],[-diam,diam]])
        vertex_x2 = vertex0.repeat(sample_tri,1,1)
        x2, c2, normal2, h2, num_each2 = gen_bd_int_points(vertex_x2,N2)
        
        #vertex0 = torch.Tensor([[-1,-1],[1,-1],[1,1],[-1,1]])
        #vertex_x1 = vertex0.repeat(sample_tri,1,1)
        #x1, c1, normal1, h1, num_each1 = gen_bd_int_points(vertex_x1,N1)
        
        
        
    return vertex_x1, x1, c1, normal1, h1, num_each1 ,  vertex_x2, x2, c2, normal2, h2, num_each2

def gen_interface_points(vertex1,vertex2,num_in):
    sample_tri = len(vertex1)
    x_in = torch.zeros(sample_tri,num_in,2)
    for p in range(sample_tri):
        xmax , xmin = (vertex2[p,:,0]).max() , (vertex2[p,:,0]).min()
        ymax , ymin = (vertex2[p,:,1]).max() , (vertex2[p,:,1]).min()
        xy0 = torch.Tensor([xmax-xmin,ymax-ymin])
        xy1 = torch.Tensor([xmin,ymin])
        x = torch.rand(num_in,2)*xy0+xy1
        for i in range(num_in):
            while (inpoly(x[i,:],vertex1[p])) or(not inpoly(x[i,:],vertex2[p])):
                x[i,:] = torch.rand(1,2)*xy0+xy1
        x_in[p] = x
    return x_in            

def interface_condition(x1,vertex_x1,x2,vertex_x2,sol_type):
    if sol_type==0:
        sample_tri,N,a = x1.shape
        u1 = torch.zeros(sample_tri,N,1)
        u2 = torch.zeros(sample_tri,N,1)
    return u1,u2    


def exact_u_interface(x1,vertex_x1,normal1,x2,vertex_x2,sol_type):
    soltype1 = sol_type//10
    soltype2 = sol_type%10
    
    u_out = exact_u(x2,vertex_x2,soltype2)
    if torch.is_tensor(normal1):
        u_jump = exact_u(x1,vertex_x2,soltype2)-exact_u(x1,vertex_x1,soltype1)
        un_jump = exact_un(x1,vertex_x2,normal1,soltype2) - exact_un(x1,vertex_x1,normal1,soltype1)
        return u_out,u_jump,un_jump
    else:
        u_in = exact_u(x1,vertex_x1,soltype1)
        return u_out,u_in
        
        




# 在调用各个子函数的时候会产生很多不需要的中间量，这里只输出用得到的，避免外部程序冗余
def main(ver_num,N,polygon_num,sol_type,a1,a2,b1,b2,problem_type,in_or_out,train,num_in,k=None):
    vertex_x, x, c, normal, h_e, num_each = gen_polygon(ver_num,N,polygon_num,a1,a2,b1,b2)    
    if problem_type == 'Poisson':
        x_in = u_in = G2_in = None
        u = G2 = None
        u = exact_u(x,vertex_x,sol_type)
        u = bd_sol_process(num_each,u)
        if train == True:
            G2 = G_bd(vertex_x, x, c, num_each,in_or_out)
            
        # 如果需要生成的内部点个数>0，说明要生成一些测试点    
        if num_in > 0:
            if in_or_out=='in':
                x_in = gen_points_in(vertex_x,num_in)
                G2_in = G_in(x_in, x, c)
                u_in0 = exact_u(x_in,vertex_x,sol_type)
                u_in = torch.zeros(polygon_num,num_in)
                for i in range(polygon_num):
                    u_in[i,:] = (u_in0[i,:]).reshape(1,-1)[0]
        return vertex_x , x , G2 , u , x_in , G2_in , u_in
    elif problem_type == 'Helmholtz':
        x_in = u_in_r = u_in_i = G2_in_r = G2_in_i = u_r = u_i = G2_r = G2_i = None
        u_r,u_i = exact_u_hel(x,vertex_x,sol_type,k)
        u_r,u_i = bd_sol_process(num_each,u_r),bd_sol_process(num_each,u_i)
        if train==True:
            G2_r, G2_i = G_hel_bd(vertex_x, x, c, num_each,in_or_out,k)
            
        if num_in>0:
            if in_or_out=='in':
                x_in = gen_points_in(vertex_x,num_in)
                G2_in_r, G2_in_i = G_Hel_in(x_in,x,c,k)
                u_in_r0,u_in_i0 = exact_u_hel(x_in,vertex_x,sol_type,k)
                u_in_r = torch.zeros(polygon_num,num_in)
                u_in_i = torch.zeros(polygon_num,num_in)
                for i in range(polygon_num):
                    u_in_r[i,:] = (u_in_r0[i,:]).reshape(1,-1)[0]
                    u_in_i[i,:] = (u_in_i0[i,:]).reshape(1,-1)[0]
        return vertex_x , x , G2_r , G2_i , u_r , u_i , x_in , G2_in_r , G2_in_i , u_in_r , u_in_i
    
            
def main_interface(problem_type,domaintype,sample_tri,vertex_num1,N1,a11,a21,b11,b21,vertex_num2,N2,a12,a22,b12,b22):
    vertex_x1, x1, c1, normal1, h1, num_each1 ,  vertex_x2, x2, c2, normal2, h2, num_each2 = gen_interface_polygon(domaintype,sample_tri,vertex_num1,N1,a11,a21,b11,b21,vertex_num2,N2,a12,a22,b12,b22)

    G_out1 =  G_bd(vertex_x2, x2, c2, num_each2,'in')
    G_out2 =  become_zero(G_single_in(x2, x1, h1),num_each2,0)

    G_in0 = G_single_bd(vertex_x1, x1, h1, num_each1)
    G_in1 = become_zero(G_in(x1,x2,c2),num_each1,1)

    G_in2 = G_bd_normal(x1,normal1,x2,c2,num_each2)
    G_in3 = G_bd_px(x1,normal1,h1,num_each1,'out')
    G_in4 = G_bd_px(x1,normal1,h1,num_each1,'in')


    #G_out1+ G_out2- G_in0- G_in1- G_in2+ G_in3+ 
    G_out2 = -G_out2
    G_in0 = -G_in0
    G_in1 = -G_in1    
    return vertex_x1, x1, c1, normal1, h1, num_each1 ,  vertex_x2, x2, c2, normal2, h2, num_each2, G_out1, G_out2,  G_in0, G_in1, G_in2, G_in3, G_in4
### 单层位势矩阵和双层位势矩阵那的正负性好像不一致 ###
### 单层位势矩阵和双层位势矩阵那的正负性好像不一致 ###
### 单层位势矩阵和双层位势矩阵那的正负性好像不一致 ###
"""
外圈 $$u(x) = -\int_{\Gamma_2}\frac{\partial G}{\partial n_y}h_2(y)ds_y-\int_{\Gamma_1} G(x,y)h_1(y)ds_y=I_1+I_2$$
内圈 $$u(x) = -\int_{\Gamma_1} G(x,y)h_1(y)ds_y = I_3$$
当 $x\in\Gamma_2$计算Dirichlet边界条件时 G_out1对应双层位势的积分矩阵 G_out2对应单层位势的积分矩阵
当 $x\in\Gamma_1$计算法向连续性gap时 G_in2 是计算$I_1$关于$n_x$求导以后的积分矩阵，G_in3是计算$I_2$关于$n_x$求导的积分矩阵，G_in4是计算$I_3$关于$n_x$求导的积分矩阵，G_in3和G_in4本质上就差了一个$\pm1/2$<br>
                计算函数在界面连续性gap时 G_in0 表示$\Gamma_1$上单层位势的积分矩阵，即计算$I_1,I_3$的，G_in1为计算$I_1$时用的积分矩阵
                 
G_single_bd G_single_in 经过测试是好的
"""
    

if __name__=='__main__':
    #整合成
    a1 = b1 = 0.8
    a2 = b2 = 0.2
    sol_type = 1
    ver_num = 6
    N = 1000
    polygon_num = 500
    
    
    ### 逐步调用
    vertex_x, x, c, normal, h_e, num_each = gen_polygon(ver_num,N,polygon_num,a1,a2,b1,b2)
    G2 = G_bd(vertex_x, x, c, num_each)
    u = exact_u(x,vertex_x,sol_type)
    u = bd_sol_process(num_each,u)
    plt.plot(vertex_x[0,:,0],vertex_x[0,:,1])
    
    ### 一次调用
    problem_type = 'Poisson'
    train = True
    vertex_x,x,G2,u,x_in,G2_in,u_in = main(ver_num,N,polygon_num,sol_type,a1,a2,b1,b2,problem_type,in_or_out,train,num_in)
    
    problem_type = 'Helmholtz'
    k = 1
    train = True
    vertex_x,x,G_r,G_i,u_r,u_i,x_in,G_in_r,G_in_i,u_in_r,u_in_i = main(ver_num,N,polygon_num,sol_type,a1,a2,b1,b2,problem_type,in_or_out,train,num_in)
