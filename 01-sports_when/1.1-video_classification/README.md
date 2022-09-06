# 算法与模型

<div align="center">
  <img src="./images/home.gif" width="550px"/><br>
</div>

## 1. 概要

本模块包含视频理解中的行为识别模型（视频分类）。视频分类和行为识别是视频理解的核心建设方向，因其训练得到的优质特征，是众多下游任务的基础。

与图像识别不同的是，行为识别任务的核心是提取时序信息。按模型结构的不同，基于RGB的行为识别方法大体上可以分为基于2D网络、基于3D网络、基于RNN以及基于Transformer结构的模型。2D网络一般会使用图像预训练模型配合时序模块提取时序信息，比如TSM等，简单高效。由于视频多一个时序维度，因此很自然的会使用3D卷积提取时序信息，比如SlowFast。3D模型的计算量一般比较大，训练迭代次数也更多一些。基于RNN的网络以视频特征作为输入，利用RNN提取时序信息，如AttentionLSTM。近期学界涌现了众多基于Transformer结构的行为识别网络，如TimeSformer。相较于卷积网络，transformer结构的网络精度更高，计算量也会大些。

详细技术方案可见各模型文档 (概览第一列包含文档链接)。

## 2. 模型概览

在UCF101-66类体育数据集上模型效果:

| 模型名称 | 骨干网络 | 测试方式 | 采样帧数 | Top-1% |
| :------: | :----------: | :----: | :----: | :----: |
| [PP-TSM](pp-tsm.md) | ResNet50 |  Uniform | 8 | 97.40 |
| [PP-TimeSformer](pp-timesformer.md) | ViT-base |  Uniform | 8 | 98.60 | 
| [SlowFast](slowfast.md) | ResNetSlowFast |  - | - | 66.80 | 
| [Attention-LSTM](attention_lstm.md) | ResNet50 |  Uniform | 8 | 95.30 | 
| [MoViNet](movinet.md) | MoViNet |  - | - | 97.90 | 

## 3. AI-Studio模型教程

- [【官方】Paddle 2.1实现视频理解优化模型 -- PP-TSM](https://aistudio.baidu.com/aistudio/projectdetail/3399656?contributionType=1)
- [【实践】CV领域的Transformer模型TimeSformer实现视频理解](https://aistudio.baidu.com/aistudio/projectdetail/3413254?contributionType=1)
