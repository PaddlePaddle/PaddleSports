# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[when（什么时间）](#1-sportswhen)，[where（什么位置）](#2-sportswhere)，[who（是谁）](#3-sportswho)，[what（发生了什么）](#4-sportswhat)，[why（为什么）](#5-sportswhy)，[how（怎么样）](#6-sportshow)。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. AI模型：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [数据集](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[SportsBenchmark](#8-sportsbenchmark)，力争能够用一个数据集来评测所有算法模型。

4. [工具集](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [应用](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。



# sports_who


&emsp; “who”模块重点分析：图像/视频中有哪几类人员，分别是谁，特定人员在整场比赛的集锦等信息：

&emsp; 1）人员分类：把图像/视频中运动员、观众、裁判、后勤工作人员进行区分；

&emsp; 2）运动员识别：识别出特定运动员，包含：人脸识别、人体识别、号码簿识别等；

&emsp; 3）运动员比赛集锦：自动生成该运动员整场比赛的视频集锦。



| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 3.who           | 3.1) 人员分类          | 运动员、裁判、观众、后勤人员              | PP-LCNetV2.md                                       | https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md                 | 王成      |
|                 | 3.2) 运动员识别         | 人脸检测                        | BlazeFace                                           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/face_detection                     | 王成      |
|                 |                    | 人脸识别                        | Dlib                                                | http://dlib.net/                                                                                            | 王成      |
|                 |                    | 基于人体的运动员识别                  |                                                     |                                                                                                             | 王成      |
|                 | 3.3) “一人一档”        | 运动员Re-ID                    | MultiSports                                         | https://github.com/MCG-NJU/MultiSports                                                                      | 王成      |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


