# -*- coding: utf-8 -*-

"""
全连接网络识别手写数字 重点在流程
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
"""
import torch
from torch import nn
from torch import optim

import numpy as np
from time import time

import get_mnist_loader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

input_size = 28*28
hidden_sizes = [128, 64] #2层隐藏层
output_size = 10 #10类数字

trainloader,valloader = get_mnist_loader.get_loaders()

#搭建全连接网络
net = nn.Sequential(nn.Linear(input_size,hidden_sizes[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[0],hidden_sizes[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[1],output_size),
                    nn.LogSoftmax(dim=1)) 

criterion = nn.NLLLoss()
# =============================================================================
# 测试基本的前向传播、误差计算和梯度反向传播
# images, labels = next(iter(trainloader))
# images = images.view(images.shape[0], -1) #-1是让程序自动判定应该是多少，这里就是28*28
# 
# logps = net(images) #log probabilities
# loss = criterion(logps, labels) #calculate the NLL loss
# print("前向传播结果", logps[0])
# print("标签", labels[0])
# print("loss: ", loss)
# print("反向传播之前没有梯度：",net[0].weight.grad)
# loss.backward()
# print("反向传播之后梯度：",net[0].weight.grad)
# =============================================================================

# 正式训练
#定义优化器
sgd_optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

#程序开始执行时间
time0 = time()

#训练15轮
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        # 前向传播
        # 首先清零梯度
        sgd_optimizer.zero_grad()
        
        output = net(images)
        loss = criterion(output, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        sgd_optimizer.step()
        
        # 累计误差
        running_loss += loss.item() # loss是一个Tensor,item()方法可以把只有一个元素的Tensor转化成标量数字
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        print("Training time of Epoch {}: {} 分钟".format(e,(time()-time0)/60))

# 测试
        
# 模型评价
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
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

# 保存模型
torch.save(net, './trained_model/my_mnist_model.pt')












