简体中文

# AGCN-2S基于骨骼的动作识别模型


## 模型简介

这篇论文做的任务是基于骨骼点进行动作识别，是对于ST-GCN的改进，然后关于ST-GCN的具体项目请详见[PPSIG：Paddlesports ST-GCN动作识别](https://aistudio.baidu.com/aistudio/projectdetail/4224807)。

[PPSIG:AGCN-2S 动作识别](https://aistudio.baidu.com/aistudio/projectdetail/4243994)项目是准备给大家逐步理解AGCN-2S的学习项目。

[AGCN-2S 封装 - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/4440380)该ai studio项目是一键运行版本。




## 数据准备

数据集请看[2021 CCF BDCI 花样滑冰选手骨骼点动作识别-训练集数据](https://aistudio.baidu.com/aistudio/datasetdetail/104925)，我把这份数据集的七分之一当作验证集，七分之六当作训练集。


## 模型训练及验证

### main.py配置参数介绍
```python
    parser.add_argument('--window_size',type=int,default=1000)
    parser.add_argument('--epoches',type=int,default=100)
    parser.add_argument('--data_file', type=str,default="/home/aistudio/data/data104925/train_data.npy")
    parser.add_argument('--label_file', type=str,default="/home/aistudio/data/data104925/train_label.npy")
    parser.add_argument('--BATCH_SIZE', type=int,default=32) 
    parser.add_argument('--load_pretrain_model', type=str,default="")
    parser.add_argument('--output_model_dir', type=str,default="model")
    parser.add_argument('--is_train',type=int,default=1)
    parser.add_argument("--output_log_dir",type=str,default="./log")
    parser.add_argument("--data_mode",type = str,default = "joint",help="joint or bone")

```
1. window_size为每个数据处理成多少帧，处理的相关代码请看stgcn_dataset.py
2. epoches为训练轮数
3. data_file为训练集中data路径
4. label_file为训练集中label路径
5. load_pretrain_model为需要导入的模型参数文件路径
6. output_model_dir为训练模型参数文件输出路径
7. is_train如果为1，则代表模型训练，如果不为1，则代表模型验证
8. output_log_dir,为log输出路径
9. data_mode为数据模式，可采用joint和bone两者，分别是关节点和骨骼
>训练的学习率优化器可以在agcn2s_main.py中自行修改，

>运行示例：!python main.py  --load_pretrain_model Gmodel_state57.pdparams --is_train 0

>Gmodel_state57.pdparams参数文件可在模型验证集准确率为0.57


## 参考论文

- [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action
Recognition](https://arxiv.org/pdf/1805.07694v3.pdf), Lei Shi1,2 Yifan Zhang1,2* Jian Cheng1,2,3 Hanqing Lu1,  2019.7