# -*- coding: utf-8 -*-
'''
输入一张图片，输出反向传播后各参数的梯度
'''
import torch
import torch.optim as optim
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import shared_model
import get_mnist_loader

ndf = 64
n_channel = 1
n_label = 10
valid_image_size = 64
batch_size = 1

# 数据集加载器
dataloader, _ = get_mnist_loader.get_loaders(valid_image_size, batch_size)
# 分类网络
netD = shared_model.netD(ndf, n_channel, n_label)
# 损失函数
c_criterion = torch.nn.NLLLoss()

    
def get_aim_grads(net):
    '''
    func:
        返回当前网络权重梯度
    params:
        net: 神经网络
    '''
    weights = []

    with torch.no_grad():
        for param in net.parameters():
            weights.append(param.grad)

    return weights
    
for i, (imgs, c_real) in enumerate(dataloader):
    print('第%d张图片'%i)
    figure = plt.figure()
    plt.imshow(imgs.numpy().squeeze(), cmap='gray_r');
    # 判别器梯度清零
    netD.zero_grad()

    # 首先用真实数据训练
    mini_batch = torch.Size([1])
    
    # 前向传播获得判别器的两部分输出
    c_output = netD(imgs)
    # 获得误差
    c_errD_real = c_criterion(c_output, c_real)
    
    # 误差反向传播
    if i==0:
        c_errD_real.backward()
    else:
        c_errD_real.backward(create_graph=True)
    
    # 提取出梯度
    # Pytorch的nn模块生成的权重似乎都是设置了retain_grad
    if i==0:
        weights = get_aim_grads(netD)
        print(weights[0])
    else:
        weights2 = get_aim_grads(netD)
        print(weights2[0])
            
    if i==1:
        break;
    i = i+1
    
loss = (weights2[0]-weights[0]).flatten()*(weights2[0]-weights[0]).flatten()
ave_loss = loss[0] 


