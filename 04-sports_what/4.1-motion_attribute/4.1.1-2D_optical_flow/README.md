# 2D_optical_flow

# 介绍
本部分主要完成了2D光流估计的相关任务，包括RAFT、FlowNet以及OpenCV实现的HS算法和LK算法

## 什么是光流
光流的概念是Gibson在1950年首先提出来的。它是空间运动物体在观察成像平面上的像素运动的瞬时速度，是利用图像序列中像素在时间域上的变化以及相邻帧之间的相关性来找到上一帧跟当前帧之间存在的对应关系，从而计算出相邻帧之间物体的运动信息的一种方法。一般而言，光流是由于场景中前景目标本身的移动、相机的运动，或者两者的共同运动所产生的。

具体来说，光流估计算法的输入是相邻的两个视频帧(shape:3×H×W)，输出的是一个光流矩阵(shape:2×H×W，两通道分别代表了水平位移和垂直位移)

## 光流估计发展
最为常用的视觉算法库OpenCV中，提供光流估计算法接口，包括稀疏光流估计算法cv2.calcOpticalFlowPyrLK()，和稠密光流估计cv2.calcOpticalFlowFarneback()。其中稀疏光流估计算法为Lucas-Kanade算法，该算法为1981年由Lucas和Kanade两位科学家提出的，最为经典也较容易理解的算法。

近几年出现了基于深度学习的光流估计算法，开山之作是FlowNet，于2015年首先使用CNN解决光流估计问题，取得了较好的结果，并且在CVPR2017上发表改进版本FlowNet2.0,成为当时State-of-the-art的方法。截止到现在，FlowNet和FlowNet2.0依然和深度学习光流估计算法中引用率最高的论文，分别引用790次和552次。随后出现了PWC、RAFT等一系列深度学习模型，并不断刷新EPE

# 贡献者
COOLGUY [AiStudio主页](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/54915)