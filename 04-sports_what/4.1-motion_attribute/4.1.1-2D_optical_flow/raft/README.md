# RAFT
## 论文地址
[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)

ECCV 2020

Zachary Teed and Jia Deng

## 网络结构
![](https://ai-studio-static-online.cdn.bcebos.com/30a5c53cc3474c9eb9f26aa468fe54e5dbd25e78d3f7428b9cb4a1c339aaef38)

RAFT由3个主要组件组成：
1. 从两个输入图像中提取每像素特征的特征编码器，以及仅从Frame中提取特征的上下文编码器。
1. 一个相关层，通过取所有特征向量对的内积，构造4D W×H×W×H相关体。4D体积的最后2维在多个尺度上汇集，以构建一组多尺度体积。
1. 一种更新操作，它通过使用当前估计值从相关体积集合中查找值来重复更新光流。


## 项目说明
本次项目分为四个部分：背景知识与论文介绍、数据集处理、模型训练、模型推理

注：当前模型使用了FlyingChairs数据集，而RAFT的实验还使用了FlyingThings数据集

raft目录下的文件夹说明：
* log存放日志
* output存放checkpoint
* core存放源码
* datasets存放数据集
* chirs_split.txt用于划分FlyingChairs的训练集和验证集
* 13724_img1.ppm 13724_img1.ppm 13724_flow.flo 是推理的样例

标准的数据集文件如下：
```
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```
## 模型训练与推理
1、目前基于深度学习的光流估计算法，都使用了大规模的数据集，包括FlyingThings、FlyingChairs等合成数据集，KITTI、Sintel等真实数据集，但是由于aistudio的存储空间、训练时间有限，所以我在本项目中**只使用了FlyingChairs数据集**，所以目前的模型在处理真实数据集时还有提升的空间。

2、最好在**V100 32G**环境训练， 如果是16G环境，需要改动一下batchsize

3、output下面的**97500**是我已经训练好的结果，可以跳过训练，使用该模型进行接下来的推理

4、目前的这个97500还可以继续优化的（小声）

```
python train.py  # 训练参数在train.py的97~114行，有详细注释
python infer.py  # 推理参数在infer.py，并且写好了图片对推理、视频推理的代码
```
测试图片如下，第一张是真实光流，第二张是推理结果，还可以继续优化的

![](https://ai-studio-static-online.cdn.bcebos.com/fd53fcb170d44d85be25a02afa7a0a62b743316f916e429a9f568bf0c03fb18e)
![](https://ai-studio-static-online.cdn.bcebos.com/b25f89cf10624a4a8d8e4004de61cd91c6baaf0454c94bbfb8b773071c54e361)

# 贡献者
COOLGUY [AiStudio主页](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/54915)
