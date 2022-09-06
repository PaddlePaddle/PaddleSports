# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[*when*](#1-when)（什么时间），[*where*](#2-where)（什么位置），[*who*](#3-who)（是谁），[*what*](#4-what)（发生了什么），[*why*](#5-why)（为什么），[*how*](#6-how)（怎么样）。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. *AI模型*：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [*数据*](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[*SportsBenchmark*](#8-benchmark)，力争能够用一个数据集来评测所有算法模型。

4. [*工具*](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [*应用*](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。




## [*what*](./04-sports_what/)

&emsp; “what”模块重点分析体育比赛画面中呈现的信息，包含：运动、语音、视觉、多模态等：

&emsp; 1）运动属性，从视频前后帧信息推断运动信息，包含2D光流以及3D场景流相关技术；

&emsp; 2）语义属性，包含：图像/视频检索识别，视频动作识别，image/video caption等；

&emsp; 3）视觉属性，包含：画质增强，超分辨率，2D转3D，3D实时交互等；

&emsp; 4）多模态属性，视觉数据与语音数据、文本数据联合分析。

| 任务              | 技术方向               | 技术细分                        | 算法模型                                                |
|-----------------|--------------------|-----------------------------|-----------------------------------------------------|
| 4.what          | 4.1) 运动属性          | 2D Optical Flow (经典算法)      | Horn-Schunck光流法                                     |
|                 |                    |                             | Lucas-Kanade光流法                                     |
|                 |                    |                             | Block-Matching光流法                                   |
|                 |                    |                             | Dual-TVL1                                           |
|                 |                    |                             | DeepFlow-v2                                         |
|                 |                    |                             | Global Patch Collider                               |
|                 |                    | 2D Optical Flow (深度学习)      | RAFT (ECCV 2020 best paper)                         |
|                 |                    |                             | FlowNet1.0                                          |
|                 |                    |                             | FlowNet2.0                                          |
|                 |                    |                             | NVIDIA SDK                                          |
|                 |                    | 3D Scene Flow               | FlowNet3D                                           |
|                 |                    |                             | Just Go with the Flow                               |
|                 |                    |                             | MotionNet                                           |
|                 |                    |                             | 2D-3D Expansion                                     |
|                 | 4.2) 语义属性          | 图像检索识别                      | PP-Lite-Shitu                                       |
|                 |                    |                             | PP-LCNetV2                                          |
|                 |                    | 视频动作识别                      | CTR-GCN                                             |
|                 |                    |                             | ST-GCN                                              |
|                 |                    |                             | AGCN                                                |
|                 |                    | Image Caption               | COCO Caption                                        |
|                 |                    |                             | Im2Text                                             |
|                 |                    | Video Caption               | ActivityNet                                         |
|                 |                    | OCR                         | PaddleOCR                                           |
|                 | 4.3) 视觉属性          | 画质增强                        | Space-Time-Aware Multi-Resolution Video Enhancement |
|                 |                    | 图像/视频去噪                     | FastDVDnet                                          |
|                 |                    | 超分辨率                        | Super Resolution                                    |
|                 |                    | 图像填补                        | Inpainting                                          |
|                 |                    | 2D转3D                       | NeRF                                                |
|                 |                    | 3D Visualization            | Maya                                                |
|                 |                    |                             | Unity                                               |
|                 |                    |                             | Unreal                                              |
|                 | 4.4) 多模态属性         | 文本+视觉                       | VideoBERT                                           |
|                 |                    |                             | VisualBERT                                          |
|                 |                    |                             |                                                     |

