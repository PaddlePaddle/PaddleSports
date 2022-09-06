简体中文

# ST-GCN基于骨骼的动作识别模型


## 模型简介

ST-GCN是AAAI 2018提出的经典的基于骨骼的行为识别模型，通过将图卷积应用在具有拓扑结构的人体骨骼数据上，使用时空图卷积提取时空特征进行行为识别，极大地提升了基于骨骼的行为识别任务精度。

[PPSIG：Paddlesports ST-GCN动作识别](https://aistudio.baidu.com/aistudio/projectdetail/4224807)这个Ai studio项目是准备给大家逐步理解ST-GCN的学习项目。

[ST-GCN 封装](https://aistudio.baidu.com/aistudio/projectdetail/4438309)这个Ai studio项目是一键运行版本。


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
```
1. window_size为每个数据处理成多少帧，处理的相关代码请看stgcn_dataset.py
2. epoches为训练轮数
3. data_file为训练集中data路径
4. label_file为训练集中label路径
5. load_pretrain_model为需要导入的模型参数文件路径
6. output_model_dir为训练模型参数文件输出路径
7. is_train如果为1，则代表模型训练，如果不为1，则代表模型验证
8. output_log_dir,为log输出路径

>训练的学习率优化器可以在stgcn_main.py中自行修改，

>运行示例：!python main.py  --load_pretrain_model Gmodel_state47.pdparams --is_train 1

>Gmodel_state47.pdparams参数文件可在模型验证集准确率为0.47


## 参考论文

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455), Sijie Yan, Yuanjun Xiong, Dahua Lin