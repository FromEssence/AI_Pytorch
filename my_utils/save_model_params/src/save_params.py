# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim

def save_current_weights(net):
    
    '''
    func: 
        拷贝一份当前网络参数的值，而非引用
    params:
        net : 需要保存当前各层权重参数的torch生成的网络
    '''
    # 保存初始梯度,不记录此操作的梯度
    # param.clone()返回的是值，不是引用
    weights = []
    with torch.no_grad():
        for param in net.parameters():
            weights.append(param.clone())
    
    return weights
