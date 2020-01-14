# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:46:25 2020
"""

'''
prepare data and train ACGAN
realated files:
    get_mnist_loader.py
'''

import torch
import torch.optim as optim
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import ACGAN
import get_mnist_loader

import random

# 设置随机数，便于比较
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
     random.seed(seed)

#setup_seed(23)

# 参数设置
valid_image_size = 64
lr = 0.0002
beta1 = 0.5 #Adam算法所需参数
batch_size = 64
nz = 100 #噪声长度
n_channel = 1 #输入图片通道数
n_label = 10
ngf = 64
ndf = 64
num_epochs = 20

model_saved_dir = './trained_model/ACGAN'
out_dir = './result/ACGAN'

# 数据集加载器
data_loader, _ = get_mnist_loader.get_loaders(valid_image_size,batch_size)

# 实例化生成器和判别器
netG = ACGAN.netG(nz, ngf, n_channel)
netD = ACGAN.netD(ndf, n_channel, n_label)

# 损失函数
s_criterion = torch.nn.BCELoss()
c_criterion = torch.nn.NLLLoss()

# 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 辅助函数 返回分类准确数
def test(predict, labels):
    
    '''
    params:
        predict:
            tensor 64*10
            对应十种分类的概率
        labels:
            tensor 64
    return:
        预测正确个数
    '''
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct

# 准备固定的用于每轮训练后可视化结果的噪声数据，将输入G中
# 生成64个[0,1]分布噪声,64*100
fixed_noise = torch.randn(batch_size, nz, 1, 1)

# 生成期待的标签
random_label = torch.randint(n_label, (batch_size,))
random_onehot = torch.zeros((batch_size, n_label))
random_onehot[torch.arange(batch_size), random_label] = 1

# 将ont-hot格式的期待标签嵌入噪声中
fixed_noise[torch.arange(batch_size), :n_label,0,0] = random_onehot[torch.arange(batch_size)]

print('用于测试的fixed label:{}'.format(random_label))

# 训练
for epoch in range(num_epochs):

    # minibatch training
    for i, (imgs, c_real) in enumerate(data_loader):
       
        # 获得此batch的数据量
        mini_batch = imgs.size()[0]

        '''先训练判别器'''
        # 判别器梯度清零
        netD.zero_grad()

        # 首先用真实数据训练
        # 真实数据的真实度标签为1
        s_real = torch.ones(mini_batch)
        # 前向传播获得判别器的两部分输出
        s_output, c_output = netD(imgs)
        # 获得误差
        s_errD_real = s_criterion(s_output, s_real)
        c_errD_real = c_criterion(c_output, c_real)
        # 总误差
        errD_real = s_errD_real+c_errD_real
        # 误差反向传播
        errD_real.backward()
        # 真假预测平均度
        s_ave = s_output.mean().item()
        # 分类准确度
        correct = test(c_output, c_real)
        c_accu = 100.0*correct/mini_batch

        # 用生成的数据训练D
        # 生成随机噪声
        # 这是我们期待的标签
        c_real = torch.randint(n_label, (mini_batch,))
        label_onehot = torch.zeros((mini_batch, n_label))
        label_onehot[torch.arange(mini_batch), c_real] = 1
        noise = torch.randn(mini_batch, nz, 1, 1)
        noise[torch.arange(mini_batch), :n_label, 0, 0] = label_onehot[torch.arange(mini_batch)]
        # 生成样本的真实度标签为0
        s_real = torch.zeros(mini_batch)

        # 生成假样本
        fake_img = netG(noise)
        # 假样本在D中前向传播
        s_output, c_output = netD(fake_img.detach()) ### fake_img并不需要记录梯度，应该将其移出计算图，并防止后面产生错误
        s_errD_fake = s_criterion(s_output, s_real)
        c_errD_fake = c_criterion(c_output, c_real)
        # 总误差
        errD_fake = s_errD_fake+c_errD_fake
        # 反向传播
        errD_fake.backward()
        # 真假预测准确率
        s_fake_ave = s_output.mean().item()
        
        errD = s_errD_real+s_errD_fake
        optimizerD.step()
        
        '''训练G'''
        netG.zero_grad()
        s_real = torch.ones(mini_batch)
        s_output,c_output = netD(fake_img) ###这里再次用到了fake_img，故上面必须保证它不被移除计算图
        s_errG = s_criterion(s_output, s_real)
        c_errG = c_criterion(c_output, c_real)

        errG = s_errG + c_errG
        errG.backward()
        D_G_z2 = s_output.mean().item()
        optimizerG.step()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Accuracy: %.4f / %.4f = %.4f'
              % (epoch, num_epochs, i, len(data_loader),
                 errD.item(), errG.item(), s_ave, s_fake_ave, D_G_z2,
                 correct, mini_batch, 100.* correct / mini_batch))

        if i % 100 == 0:
            vutils.save_image(imgs,
                    '%s/real_samples.png' % out_dir)
            #fake = netG(fixed_cat)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (out_dir, epoch))

        
        # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (model_saved_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (model_saved_dir, epoch))
