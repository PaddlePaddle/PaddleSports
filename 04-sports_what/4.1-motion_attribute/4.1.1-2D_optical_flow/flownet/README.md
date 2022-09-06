# FlowNet 1.0

## 论文地址
[FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852v2https://arxiv.org/abs/1504.06852v2)
ICCV 2015

## 网络结构
![](https://ai-studio-static-online.cdn.bcebos.com/a219bb9e356f4615a53f116eb79ebada5b20d12d939940c486284ffcfb3839ea)

如上图所示就是FlowNet神经网络的大体思路。输入为两张图像，他们分别是第 t 帧以及第 t+1 帧的图像，它们首先通过一个由卷积层组成的收缩部分，用以提取各自的特征矩阵，但是这样会使图片缩小，因此需要再通过一个扩大层，将其扩展到原图大小，进行光流预测，和Unet是比较相似的。

![](https://ai-studio-static-online.cdn.bcebos.com/c0723f58a65742da8efc9639be63b5966785890e36394330ae277ab9947b0ef0)

上图为收缩部分的网络结构，直接将两个3维图像叠加成6维矩阵输入到网络中。（除了这种方案，还有一个FlowNetCorr方案，这种方案在本项目的后续版本中将会更新）

![](https://ai-studio-static-online.cdn.bcebos.com/49ae13f018e14ff5a2890043f11e3a303b9ea85449334d52a00110625277bab8)

上图为放大部分的网络结构，主要是由逆卷积组成。


## 文件说明
work目录下的文件夹说明：
* log存放日志
* output存放checkpoint
* core和models存放源码
* datasets存放数据集
* chirs_split.txt用于划分FlyingChairs的训练集和验证集

数据集依然使用了FlyingChairs，具体介绍可以参考这个项目[RAFT 光流估计模型](https://aistudio.baidu.com/aistudio/projectdetail/4306294)


## 模型训练及验证
```
# 具体参数的调整可以在train.py的58~68行进行修改
# 由于本身就是光流估计的初代模型，且发布时间久远，flownet的启发意义大于模型的效果，不必特意追求该模型的拟合效果
python train.py 

```
# 贡献者
COOLGUY [AiStudio主页](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/54915)



