import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np 
from scipy import interpolate


# 本文件是raft模型中的特征提取和上下文提取模块，模块主要使用了残差卷积


class ResidualBlock(nn.Layer):
    def __init__(self, in_planes, planes, norm_fn='batch', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes,kernel_size=3, padding=1, stride=stride, weight_attr=nn.initializer.KaimingNormal())
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, padding=1, weight_attr=nn.initializer.KaimingNormal())
        self.relu = nn.ReLU()  # in_planes=True
 
        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2D(planes, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
            self.norm2 = nn.BatchNorm2D(planes, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
            if not stride == 1:
                self.norm3 = nn.BatchNorm2D(planes, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2D(planes, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
            self.norm2 = nn.InstanceNorm2D(planes, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2D(planes, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
        
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_planes, planes, kernel_size=1, stride=stride, weight_attr=nn.initializer.KaimingNormal()), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)


class BasicEncoder(nn.Layer):
    def __init__(self, output_dim=128, norm_fn='batch'):
        # 可以自行加入dropout
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        
        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2D(64, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2D(64, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)))
        
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, weight_attr=nn.initializer.KaimingNormal())
        self.relu1 = nn.ReLU()
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2D(128, output_dim, kernel_size=1, weight_attr=nn.initializer.KaimingNormal())

        # self.dropout = None
        
        # 初始化
        # for m in self.children():
        #     print(m)
        #     if isinstance(m, nn.Conv2D):
        #         x = "原版code是在此处再遍历进行初始化,使用的是KaimingNormal的fan_out," \
        #             "paddle只有fan_in, 所以使用了fan_in, 并且把初始化的步骤放在了conv声明的位置"
        #     elif isinstance(m, (nn.BatchNorm2D, nn.InstanceNorm2D)): 
        #         # 此处是对BN和IN都进行了初始化，d但是总是报错，提示没有weight这个属性，我深究这个细节意义不大，于是我把初始化都放在了声明的地方
        #         # nn.initializer.constant(m.weight, 1)
        #         # nn.initializer.constant(m.bias, 0)
        #         if m.weight is not None:
        #             nn.initializer.constant(m.weight, 1)
        #         if m.bias is not None:
        #             nn.initializer.constant(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layer = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layer)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)

        if is_list:
            batch_dim = x[0].shape[0]
            # print(x[0].shape)
            x = paddle.concat(x, axis=0)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.conv2(x)
        
        if is_list:
            x = paddle.split(x, [batch_dim, batch_dim], axis=0)
        return x
