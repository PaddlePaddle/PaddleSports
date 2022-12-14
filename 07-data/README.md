# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[*when*](#1-when)（什么时间），[*where*](#2-where)（什么位置），[*who*](#3-who)（是谁），[*what*](#4-what)（发生了什么），[*why*](#5-why)（为什么），[*how*](#6-how)（怎么样）。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. *AI模型*：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [*数据*](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[*SportsBenchmark*](#8-benchmark)，力争能够用一个数据集来评测所有算法模型。

4. [*工具*](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [*应用*](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。


## [*data*](./07-data/)

&emsp; “data”模块重点梳理生成训练数据的6种主流方式：

&emsp; 1）人工标注：已标注的公开数据集，用于网络训练；

&emsp; 2）迁移学习：未标注的大量数据，做非监督学习和迁移学习；

&emsp; 3）合成数据：2D图像直接编辑，copy-paste的方式合成训练数据；

&emsp; 4）合成数据：3D模型渲染生成2D数据以及标注信息；

&emsp; 5）合成数据：3D模型部件指导的2D图像编辑；

&emsp; 6）合成数据：GAN系列网络模型合成训练数据。

| 任务              | 技术方向               | 技术细分                        | 算法模型                                                |
|-----------------|--------------------|-----------------------------|-----------------------------------------------------|
| 7.data          | 7.1) 已标注的数据集       |                             |                                                     |
|                 | 7.2) 未标注的数据集       |                             |                                                     |
|                 | 7.3) 2D Copy-Paste |                             |                                                     |
|                 | 7.4) 3D Rendering  |                             |                                                     |
|                 | 7.5) 3D-2D Editing |                             |                                                     |
|                 | 7.6) GAN           |                             |                                                     |
|                 |                    |                             |                                                     |


