# -*- coding: utf-8 -*-

'''
 train the model for MNIST classification  
'''
import torch
import torch.optim as optim
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

import shared_model
import get_mnist_loader

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
            weights.append(param)

    return weights

def validata_model(net, trainloder):
    # 模型评价
    correct_count, all_count = 0, 0
    for images,labels in trainloder:
      for i in range(len(labels)):
        img = images[i].view(1,1,64,64)
        with torch.no_grad():
            logps = net(img)
    
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1
    
    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))    
    

ndf = 64
n_channel = 1
n_label = 10
valid_image_size = 64
batch_size = 64
n_epoch = 10

# 数据集加载器
dataloader, valloader = get_mnist_loader.get_loaders(valid_image_size, batch_size)
# 分类网络
netD = shared_model.netD(ndf, n_channel, n_label)
# 损失函数
c_criterion = torch.nn.NLLLoss()
# 优化方法
sgd_optimizer = optim.SGD(netD.parameters(), lr=0.005, momentum=0.9)

for e in range(n_epoch):
    running_loss = 0
    for images, labels in dataloader:
        #print('here')
        # 前向传播
        # 首先清零梯度
        sgd_optimizer.zero_grad()
        
        output = netD(images)
        
        # print(output, labels)
        loss = c_criterion(output, labels)
        
        print('loss', loss)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        sgd_optimizer.step()
        
        # 累计误差
        running_loss += loss.item() # loss是一个Tensor,item()方法可以把只有一个元素的Tensor转化成标量数字
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(dataloader)))      
    
    torch.save(netD.state_dict(), './trained_model/MNIST/netD_epoch_%d.pth' % (e))
    print("第{}轮训练后训练集误差".format(e))
    validata_model(netD,valloader)
    
