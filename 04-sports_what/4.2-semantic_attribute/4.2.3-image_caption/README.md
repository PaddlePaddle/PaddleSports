# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[when（什么时间）](#1-sportswhen)，[where（什么位置）](#2-sportswhere)，[who（是谁）](#3-sportswho)，[what（发生了什么）](#4-sportswhat)，[why（为什么）](#5-sportswhy)，[how（怎么样）](#6-sportshow)。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. AI模型：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [数据集](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[SportsBenchmark](#8-sportsbenchmark)，力争能够用一个数据集来评测所有算法模型。

4. [工具集](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [应用](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。




# sports_what

&emsp; “what”模块重点分析体育比赛画面中呈现的信息，包含：运动、语音、视觉、多模态等：

&emsp; 1）运动属性，从视频前后帧信息推断运动信息，包含2D光流以及3D场景流相关技术；

&emsp; 2）语义属性，包含：图像/视频检索识别，视频动作识别，image/video caption等；

&emsp; 3）视觉属性，包含：画质增强，超分辨率，2D转3D，3D实时交互等；

&emsp; 4）多模态属性，视觉数据与语音数据、文本数据联合分析。


| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 4.what          | 4.1) 运动属性          | 2D Optical Flow (经典算法)      | Horn-Schunck光流法                                     | opencv.CalcOpticalFlowHS                                                                                    | 张熙瑞     |
|                 |                    |                             | Lucas-Kanade光流法                                     | cv::optflow::calcOpticalFlowSparseToDense()                                                                 | 张熙瑞     |
|                 |                    |                             | Block-Matching光流法                                   | opencv.calcOpticalFlowBM                                                                                    | 张熙瑞     |
|                 |                    |                             | Dual-TVL1                                           | https://docs.opencv.org/4.5.5/dc/d4d/classcv_1_1optflow_1_1DualTVL1OpticalFlow.html                         | 张熙瑞     |
|                 |                    |                             | DeepFlow-v2                                         | http://lear.inrialpes.fr/src/deepflow/                                                                      | 张熙瑞     |
|                 |                    |                             | Global Patch Collider                               | https://docs.opencv.org/4.5.5/d8/dc5/sparse__matching__gpc_8hpp.html                                        | 张熙瑞     |
|                 |                    | 2D Optical Flow (深度学习)      | RAFT (ECCV 2020 best paper)                         | https://github.com/princeton-vl/RAFT                                                                        | 张熙瑞     |
|                 |                    |                             | FlowNet2.0                                          | https://github.com/NVIDIA/flownet2-pytorch                                                                  | 张熙瑞     |
|                 |                    |                             | NVIDIA SDK                                          | https://developer.nvidia.com/opticalflow-sdk                                                                | 张熙瑞     |
|                 |                    | 3D Scene Flow               | FlowNet3D                                           | https://github.com/xingyul/flownet3d                                                                        | 张熙瑞     |
|                 |                    |                             | Just Go with the Flow                               | https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation                     | 张熙瑞     |
|                 |                    |                             | MotionNet                                           | https://www.merl.com/research/?research=license-request&sw=MotionNet                                        | 张熙瑞     |
|                 |                    |                             | 2D-3D Expansion                                     | https://github.com/gengshan-y/expansion                                                                     | 张熙瑞     |
|                 | 4.2) 语义属性          | 图像检索识别                      | PP-Lite-Shitu                                       | https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/deploy/lite_shitu                               | 洪力      |
|                 |                    |                             | PP-LCNetV2                                          | https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md                 | 洪力      |
|                 |                    | 视频动作识别                      | CTR-GCN                                             | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/ctrgcn.md         | 洪力      |
|                 |                    |                             | ST-GCN                                              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md          | 洪力      |
|                 |                    |                             | AGCN                                                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/agcn.md           | 洪力      |
|                 |                    | Image Caption               | COCO Caption                                        | https://github.com/tylin/coco-caption                                                                       | 王庆忠     |
|                 |                    |                             | Im2Text                                             | https://www.cs.virginia.edu/~vicente/sbucaptions/                                                           | 王庆忠     |
|                 |                    | Video Caption               | ActivityNet                                         | http://activity-net.org/challenges/2017/captioning.html                                                     | 王庆忠     |
|                 | 4.3) 视觉属性          | 画质增强                        | Space-Time-Aware Multi-Resolution Video Enhancement | https://github.com/alterzero/STARnet                                                                        | 卢飞翔     |
|                 |                    | 图像/视频去噪                     | FastDVDnet                                          | https://github.com/m-tassano/fastdvdnet                                                                     | 卢飞翔     |
|                 |                    | 超分辨率                        | Super Resolution                                    |                                                                                                             | 卢飞翔     |
|                 |                    | 图像填补                        | Inpainting                                          |                                                                                                             | 卢飞翔     |
|                 |                    | 2D转3D                       | NeRF                                                |                                                                                                             | 卢飞翔     |
|                 |                    | 3D Visualization            | Maya                                                |                                                                                                             | 卢飞翔     |
|                 |                    |                             | Unity                                               |                                                                                                             | 卢飞翔     |
|                 |                    |                             | Unreal                                              |                                                                                                             | 卢飞翔     |
|                 | 4.4) 多模态属性         | 文本+视觉                       | VideoBERT                                           |                                                                                                             | 王庆忠     |
|                 |                    |                             | VisualBERT                                          |                                                                                                             | 王庆忠     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |



