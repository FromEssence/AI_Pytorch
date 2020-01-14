# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:16:55 2020

"""
import torch
from torchvision import datasets, transforms

def get_loaders(scale_size=0, bs=64):
    """
    params:
        bs : batch_size
    return: [trainloader, valloader]
    usage:
        #查看数据格式
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        
        print(images.shape, labels.shape)
        
        #1*28*28变成28*28
        plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
        
        多个图像概览
        figure = plt.figure()
        num_of_images = 60
        for index in range(1, num_of_images + 1):
            plt.subplot(6, 10, index)
            plt.axis('off')
            plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    """
    #转化为[0,1]区间,并且规范化，
    transform = None
    if scale_size != 0:
        transform = transforms.Compose([transforms.Scale(scale_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    
    #使用Pytorch预设置的MNIST加载器
    trainset = datasets.MNIST('./datasets/mnist', download=False, train=True, transform=transform)
    valset = datasets.MNIST('./datasets/mnist', download=False, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=True)
    
    return [trainloader, valloader]