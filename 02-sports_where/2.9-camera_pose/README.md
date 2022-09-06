# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[*when*](#1-when)（什么时间），[*where*](#2-where)（什么位置），[*who*](#3-who)（是谁），[*what*](#4-what)（发生了什么），[*why*](#5-why)（为什么），[*how*](#6-how)（怎么样）。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. *AI模型*：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [*数据*](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[*SportsBenchmark*](#8-benchmark)，力争能够用一个数据集来评测所有算法模型。

4. [*工具*](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [*应用*](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。




## [*where*](./02-sports_where/)

&emsp; “where”模块重点分析：前景（运动员）、背景（场馆）、相机，这三类对象的位置/位姿的信息：

&emsp; 1）运动员整体位姿：图像/视频中运动员的2D/3D定位，包含：2D/3D检测、2D分割、2D/3D跟踪等；

&emsp; 2）运动员局部位姿：运动员的骨骼姿态的分析，从粗粒度到细粒度，包含：2D骨骼关键点、2D骨骼姿态、3D骨骼姿态、2D-3D稠密映射、3D人体重建、3D人体动画等；

&emsp; 3）背景3D重建：利用多维传感器数据，1比1重建场馆的3D信息，相关技术包含：Simultaneous Localization and Mapping (SLAM)、Structure-from-Motion (SfM) 等；

&emsp; 4）相机6-DoF位姿：恢复相机的6-DoF位姿（位置xyz，旋转αβγ），有经典的PNP算法，以及深度模型算法。

| 任务              | 技术方向               | 技术细分                        | 算法模型                                                |
|-----------------|--------------------|-----------------------------|-----------------------------------------------------|
| 2.where         | 2.1) 2D检测          | 一阶段通用目标检测                   | PP-YOLOE                                            |
|                 |                    |                             | PP-PicoDet                                          |
|                 |                    | 二阶段通用目标检测                   | Faster-RCNN                                         |
|                 |                    | 人体检测分析                      | PP-Human2.0                                         |
|                 |                    |                             | PP-Pedestrian                                       |
|                 |                    | 水花/足球/篮球等小目标检测              | FPN，PP-YOLOE                                        |
|                 |                    |                             |                                                     |
|                 | 2.2) 2D分割          | 前景对象/背景分割                   | Mask-RCNN                                           |
|                 |                    |                             | SOLOv2                                              |
|                 |                    |                             | PP-LiteSeg                                          |
|                 |                    |                             | DeepLabV3P                                          |
|                 |                    | 交互式分割                       | EISeg                                               |
|                 |                    | 人体分割                        | PP-HumanSeg                                         |
|                 |                    | 人体毛发级精准分割                   | Matting                                             |
|                 |                    |                             | Human Matting                                       |
|                 |                    | 视频目标分割                      | CFBI                                                |
|                 |                    |                             | MA-Net                                              |
|                 |                    | 视频运动物体分割                    | Motion Segmentation                                 |
|                 |                    | 视频人体分割 Video Matting        | BackgroundMattingV2                                 |
|                 |                    |                             |                                                     |
|                 | 2.3) 2D跟踪          | 人体跟踪                        | ByteTrack                                           |
|                 |                    | 运动轨迹                        | PP-Tracking                                         |
|                 |                    |                             |                                                     |
|                 | 2.4) 2D骨骼          | Top-Down                    | PP-TinyPose                                         |
|                 |                    |                             | HR-Net                                              |
|                 |                    | Bottom-Up                   | OpenPose                                            |
|                 |                    |                             | MoveNet                                             |
|                 |                    |                             |                                                     |
|                 | 2.5) 3D骨骼          | 单目                          | PP-TinyPose3D                                       |
|                 |                    |                             | Position-based                                      |
|                 |                    |                             | Angle-based                                         |
|                 |                    |                             | 2D + Depth-based                                    |
|                 |                    |                             | 2D + IK                                             |
|                 |                    | 多目                          | Calibration                                         |
|                 |                    |                             | Fusion                                              |
|                 |                    | 深度相机                        | Kinect 3D Tracking                                  |
|                 |                    |                             |                                                     |
|                 | 2.6) 2D/3D稠密映射     | 2D-2D Dense Correspondences | DeepMatching                                        |
|                 |                    | 2D-3D Dense Correspondences | DensePose                                           |
|                 |                    |                             |                                                     |
|                 | 2.7) 3D人体重建        | Template Model              | SMPL                                                |
|                 |                    |                             | VIBE                                                |
|                 |                    |                             | PyMaf                                               |
|                 |                    |                             |                                                     |
|                 | 2.8) SLAM          | 静态                          | 单目 ORB-SLAM...                                      |
|                 |                    |                             | 深度 KinectFusion...                                  |
|                 |                    |                             | 激光 LOAM                                             |
|                 |                    | 动态                          | DynamicFusion                                       |
|                 |                    |                             | DynSLAM                                             |
|                 |                    |                             |                                                     |
|                 | 2.9) 相机6-DoF定位     | 内参                          | 张氏标定法                                               |
|                 |                    | 外参                          | 单张图像 PNP                                            |
|                 |                    |                             | 多张图像 SfM, SLAM                                      |
|                 |                    |                             |                                                     |

