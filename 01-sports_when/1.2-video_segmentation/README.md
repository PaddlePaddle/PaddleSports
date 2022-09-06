# 算法与模型

![image](https://raw.githubusercontent.com/FeixiangLu/PaddleSports/main/01-sports_when/1.2-video_segmentation/images/predict.png)

## 1. 概要

本模块包含视频理解中主流的动作定位模型[BMN](BMN.md)和冠军模型[TCANet++](TCANet++.md), 详细技术方案见模型文档。

## 2. 模型概览

在ActivityNet1.3数据集上评估精度如下:

| 模型名称 | AR@1 | AR@5 | AR@10 | AR@100 | AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| [BMN](BMN.md) | 33.26 | 49.48 | 56.86 | 75.19 | 67.23% |

在乒乓球比赛转播视频数据集上评估精度如下:

| 模型名称 | 骨干网络 | AUC |
| :------: | :----------: | :----: |
| [BMN](BMN.md) | PP-TSM | 19.30% |
| [TCANet++](TCANet++.md) | PP-TSM | 48.51% | 

## 3. AI-Studio模型教程

- [BMN视频动作定位](https://aistudio.baidu.com/aistudio/projectdetail/2250674)
- [TCANet++冠军模型](https://aistudio.baidu.com/aistudio/projectdetail/3545680)
