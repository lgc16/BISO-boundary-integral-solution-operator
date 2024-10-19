# 此文件夹收录了边界到解映射的一些模块

模块包括
- 生成数据模块 GenData.py
- 生成网络模块 Net_architecture.py

## 模块介绍

### GenData

- gen_polygon(vertex_num,N,sample_tri,a1,a2,b1,b2)
生成多边形、多边形上积分点、和计算积分权重时要用到的量
输入介绍：
         vertex_num  顶点数
         N           积分积分点数
         sample_tri  生成的多边形个数
         a1,a2,b1,b2 生成多边形时控制随机性的参数
返回值：vertex_x, x, c, normal, h, num_each
         vertex_x sample_tri*vertex_num*2 的tensor，表示生成的多边形顶点坐标
         x        sample_tri*N*2 的tensor，表示生成的多边形上积分点坐标
         c        sample_tri*N*2 的tensor，由normal和h确定，用于之后生成积分权重
         normal   sample_tri*N*2 的tensor，每一段上的外法向量（注意不是顶点）
         h        sample_tri*N*1 的tensor，相邻积分点之间的距离  
生成多边形方法介绍：
        1、随机生成vertex_num 个a1-a2之间的值t_1,t_2,...
        2、另theta_i = (t_1+...+t_i)/sum(t)*2*pi 作为极角
        3、随机生成vertex_num 个b1-b2之间的值r_1,r_2,... 作为极径
         
- G_bd(vertex_x, x, c, num_each)
- gen_points_in(vertex_x,num_in)
- G_in(x_in, x, c)
- bd_sol_process(num_each,u)
- gen_point_in(vertex,num_in)
- inpoly(x,vertex)
- exact_u(x,vertex_x,sol_type)

- 待写的函数：写一个main函数整合一下生成训练点、测试点、积分矩阵啥的，免的主程序中这一块臃肿
            Helmholtz方程的积分权重矩阵

### Net_architecture
- PointNet()
仿照PointNet写的一个网络，理论上可以处理任意输入点数，但是现在把输入点数N=1000 写成了hardcode，修改的话需要把maxpooling从init中移到forward中



## 更新信息
