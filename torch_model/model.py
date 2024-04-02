import torch.nn as nn
import torch.nn.functional as F
import torch
import tsk
import argparse
import pandas as pd
from scipy.io import loadmat
## 定义参数初始化函数
def weights_init_normal(m):                                    
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字. 
    if classname.find("Conv") != -1:                        ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.


# 定义残差块，用于生成器中实现特征的恒等映射，增强网络的学习能力。
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # 残差块内部的结构：包含填充、卷积、归一化和ReLU激活函数
        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),  # 使用反射填充来减少边缘效应
            nn.Conv1d(in_features, in_features, 3),  # 1D卷积操作
            nn.InstanceNorm1d(in_features),  # 实例归一化，用于处理每个样本
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.ReflectionPad1d(1),  # 再次使用反射填充
            nn.Conv1d(in_features, in_features, 3),  # 另一次1D卷积操作
            nn.InstanceNorm1d(in_features)  # 再次实例归一化
        )

    def forward(self, x):
        return x + self.block(x)  # 将输入加上残差块处理后的输出，实现恒等映射

# 定义生成器网络，采用ResNet架构进行特征转换
class GeneratorResNet(nn.Module):
    def __init__(self, input_dim=640, output_dim=640, n_rules=50):
        super(GeneratorResNet, self).__init__()
        # 初始化TSK模型作为生成器的核心，使用适当的规则数
        self.tsk = tsk.TSK(in_dim=input_dim, out_dim=(output_dim,), n_rules=n_rules, antecedent="tsk")
    def forward(self, x):
        # 直接调用TSK模型进行前向传播
        self.tsk.init_model(x, y=None, scale=1, std=0.2, method="cluster", sigma=None, cluster_kwargs=None, eps=1e-8)
        output, _, _, _, _ = self.tsk(x)
        return output
    

class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()

        # input_shape改为一维数据长度，例如640维特征向量
        channels = 1  # 对于一维数据，通道数设置为1

        # 定义鉴别器块，用于构建鉴别器网络
        def discriminator_block(in_filters, out_filters, normalize=True):
            """构造每个鉴别器块的下采样层"""
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 构建鉴别器模型，通过一系列鉴别器块进行特征下采样
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),  # 第一层不进行归一化
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # 由于是一维数据，最后一层的padding和卷积操作需相应调整
            nn.Conv1d(512, 1, 4, padding=1)  # 输出层卷积，将特征映射到单个值
        )

        # 计算输出形状，这对于一维数据来说通常是长度的一维表示
        # 由于这里是鉴别器，主要关注的是输出的判别结果，具体输出形状依模型结构而定
        self.output_shape = (1, input_length // 2 ** 4)

    def forward(self, x):
        return self.model(x)


##读取数据，day_0：0725，day_n：0819
file_path_day_0 = 'D:/learning/脑机接口资料/active_learning/0725.xls'
file_path_day_n = 'D:/learning/脑机接口资料/active_learning/0819.xls'
data_day_0 = pd.read_excel(file_path_day_0, header=None)
data_day_n = pd.read_excel(file_path_day_n, header=None)
data_day_0 = torch.tensor(data_day_0.values)
data_day_n = torch.tensor(data_day_n.values)
# 设置设备

G_AB = GeneratorResNet(input_dim=640, output_dim=640, n_rules=50)
D_B = Discriminator(input_length=640)
outputdata = G_AB.forward(data_day_0)
dis = D_B.forward(data_day_0)
print(outputdata)
print(dis)