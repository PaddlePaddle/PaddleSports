# 算法与模型

![image](https://raw.githubusercontent.com/FeixiangLu/PaddleSports/main/01-sports_when/1.3-video_parsing/images/action_example.png)

## 1. 概要

本模块包含视频理解中主流的动作分割模型[ASRF](asrf.md), [MS-TCN](mstcn.md), 和[CFBI](cfbi.md), 详细技术方案见模型文档。

## 2. 模型概览

在Breakfast数据集下评估精度如下(采用4折交验证):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ASRF | 66.1% | 71.9% | 73.3% | 67.9% | 55.7% |
| MS-TCN | 65.2% | 61.5% | 53.7% | 49.2% | 38.8% |


在50salads数据集下评估精度如下(采用5折交验证):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ASRF | 81.6% | 75.8% | 83.0% | 81.5% | 74.8% |
| MS-TCN | 81.1% | 71.5% | 77.9% | 75.5% | 66.5% |

在gtea数据集下评估精度如下(采用4折交验证):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ASRF | 77.1% | 83.3% | 88.9% | 87.5% | 79.1% |
| MS-TCN | 76.9% | 81.8% | 86.4% | 84.7% | 74.8% |

在DAVIS数据集测试精度:

| Model | J&F-Mean | J-Mean | J-Recall | J-Decay | F-Mean | F-Recall | F-Decay |
| :------: | :------: | :-----: | :----: | :----: | :----: | :----: | :----: |
| CFBI | 0.823 | 0.793 | 0.885 | 0.083 | 0.852 | 0.932 | 0.100 |


## 3. AI-Studio模型教程

- [《产业级视频技术与应用案例》系列课程(包含ASRF、MS-TCN、CFBI算法介绍)](https://aistudio.baidu.com/aistudio/course/introduce/6742)


