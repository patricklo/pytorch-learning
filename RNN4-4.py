import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
'''读取数据
- 分别构建训练集和测试集（验证集）
- DataLoader来迭代取数据
'''

#定义参数
input_size=28 #输入图像的尺寸 28*28
num_classes = 10 #标签的总类数
num_epochs = 3 #训练的总循环周期
batch_size=64 #一个批次的大小， 64张图片
#训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform = transforms.ToTensor(), download=True)
#测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform = transforms.ToTensor())
#构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print('test')

'''卷积网络模块的构建
- 一般卷积层/relu层/池化层可以定成一个套餐
- 注意卷积最后结果还是一个特征图，需要把图转换成向量才能做分类或者回归任务
'''


class MY_CNN(nn.Module):
    def __init__(self):
        super(MY_CNN, self).__init__()
        '''
        PyTorch中要求卷积输入维度为channel first概念，也就是说一张28*28的黑白照片（channel为1)-(28,28,1), 在pytorch要转换成（1，28，28）的维度来作为输入.
        因此后续如果读入维度为(28,28,1)需要自己转换为(1,28,28)

        ## 2d卷积，也就是2d图像中的height * weight， 3d卷积对应的是视频数据，height*weight*time

        H2 = (H1 - Fh + 2P)/S + 1 = (28-5+2*2)/1 + 1 = 28
        W2 = (W1 - Fw + 2P)/S + 1= (28-5+2*2)/1 + 1 = 28

        输入维度(1, 28, 28) ->输出维度(16(out_channels) * H2 * W2) = (16, 28, 28)

        '''
        self.conv1 = nn.Sequential(  ## <-----输入大小(1,28,28)，其实就是(28,28,1)
            nn.Conv2d(  ## 2d卷积，也就是2d图像中的height * weight， 3d卷积对应的是视频数据，height*weight*time
                in_channels=1,  # 灰度图, 如果是彩色照片RGB那channel是3
                out_channels=16,  # 需要输出多少个特征图 （4*4) - 其实就是卷积核（权重/weight)的个数
                kernel_size=5,  # 每个卷积核大小-weight权重的大小 2d卷积对应的是5*5? Fw * Fh = 5 *５
                stride=1,  # 步长　S
                padding=2),  # padding: P, 如果希望卷积后大小和原来一样，padding=(kernel_size-1)/2, if stride = 1_
            # 输出的特征图为（16,28,28)->(28,28,16)  - 16是通道数channel

            nn.ReLU(),  # ReLU层/sigmoid层，做完一次特征提取后，都要做一下非线性映射（relu)
            nn.MaxPool2d(kernel_size=2),  # 进行池化操作,其实是压缩操作。 (2*2区域）----> 输出为：(16, 28/kernel_size=14, 28/kernel_size=14)
        )
        self.conv2 = nn.Sequential(  ##输入为：上面一层的输入： (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  ##输出为：(32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  ##输出为：(32, 7, 7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),  # 输出（64，14，14）
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2) #输出为：（64，7，7）
        )

        '''
        ##全连接：从上面的输出特征图（64＊７＊７）-> wx+b　->　得到１０分类的概率结果 

        特征图（64＊７＊７）->拉长为向量（长度为：64*7*7=3136）-> wx+b -> 10分类的概率结果

        拉长为向量: (x = x.view(x.size(0), -1) )
        '''
        self.out = nn.Linear(64 * 7 * 7, 10)  ##全连接：从上面的输出（６４＊７＊７）　->　得到１０分类的概率结果

    def forward(self, x):  ##X有4个维度： (batch_size, channel, height, weight) -> (batch_size, 64, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        '''
        x.view(x.size(0), -1)

        x.size(0)=batch_size, 
        -1:即自动计算另一个维度大小。

        跟reshape操作一样，flatten的操作，将(batch_size, 64, 7, 7)的维度转换为：(batch_size, 64*7*7) 
        '''
        x = x.view(x.size(0), -1)
        output = self.out(x)  # 全连接，得出结果。
        return output

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]  #得结果当中，值最大的值
    rights = pred.eq(labels.data.view_as(pred)).sum()  #看pred值 跟labels是不是相等的。
    return rights, len(labels)