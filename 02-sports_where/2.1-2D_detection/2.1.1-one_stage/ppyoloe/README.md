简体中文 | [English](https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.4/configs/ppyoloe/README.md)

# 使用PP-YOLOE/视频中的运动员进行检测、定位
## 简介
PP-YOLOE是基于PP-YOLOv2的卓越的单阶段Anchor-free模型，超越了多种流行的yolo模型。
PP-YOLOE有一系列的模型，即s/m/l/x，可以通过width multiplier和depth multiplier配置。
PP-YOLOE避免使用诸如deformable convolution或者matrix nms之类的特殊算子，以使其能轻松地部署在多种多样的硬件上。
更多细节可以参考我们的[report](https://arxiv.org/abs/2203.16250)。

![](https://ai-studio-static-online.cdn.bcebos.com/298c772e8ead4ce4bdabae5313af2e7985cb26d8162a4518a3ea5de864d74466)

PP-YOLOE-l在COCO test-dev2017达到了51.4的mAP, 同时其速度在Tesla V100上达到了78.1 
FPS。PP-YOLOE-s/m/x同样具有卓越的精度速度性价比, 其精度速度可以在[模型库](#模型库)中找到。

PP-YOLOE由以下方法组成
- 可扩展的backbone和neck
- [Task Alignment Learning](https://arxiv.org/abs/2108.07755)
- Efficient Task-aligned head with [DFL](https://arxiv.org/abs/2006.04388)和[VFL](https://arxiv.org/abs/2008.13367)
- [SiLU激活函数](https://arxiv.org/abs/1710.05941)

## 模型库
|          模型           | GPU个数 | 每GPU图片个数 |  骨干网络  | 输入尺寸 | Box AP<sup>val</sup> | Box AP<sup>test</sup> | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 | 配置文件  |
|:------------------------:|:-------:|:--------:|:----------:| :-------:| :------------------: | :-------------------: |:---------:|:--------:|:---------------:| :---------------------: | :------: | :------: |
| PP-YOLOE-s                  |     8      |    32    | cspresnet-s |     640     |       42.7        |        43.1         |   7.93    |  17.36   |       208.3 |          333.3          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)                   |
| PP-YOLOE-m                  |     8      |    28    | cspresnet-m |     640     |       48.6        |        48.9         |   23.43   |  49.91   |   123.4   |  208.3   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml)                   |
| PP-YOLOE-l                  |     8      |    20    | cspresnet-l |     640     |       50.9        |        51.4         |   52.20   |  110.07  |   78.1    |  149.2   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)                   |
| PP-YOLOE-x                  |     8      |    16    | cspresnet-x |     640     |       51.9        |        52.2         |   98.42   |  206.59  |   45.0    |   95.2   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml)                   |

**注意:**

- PP-YOLOE模型使用COCO数据集中train2017作为训练集，使用val2017和test-dev2017作为测试集，Box AP<sup>test</sup>为`mAP(IoU=0.5:0.95)`评估结果。
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- PP-YOLOE模型推理速度测试采用单卡V100，batch size=1进行测试，使用**CUDA 10.2**, **CUDNN 7.6.5**，TensorRT推理速度测试使用**TensorRT 6.0.1.8**。
- 参考[速度测试](#速度测试)以复现PP-YOLOE推理速度测试结果。
- 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

* 更多详情信息可以参考[连接](https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.4/configs/ppyoloe/README_cn.md) 

## 使用说明

* 如果您想了解具体细节可以参考AIstudio-[使用Picodet/PP-yoloe检测运动员和足球](https://aistudio.baidu.com/aistudio/projectdetail/4479428?contributionType=1&sUid=206265&shared=1&ts=1661954440536)


### 训练

* 多卡训练：详情信息可以参考[连接](https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.4/configs/ppyoloe/README_cn.md#%E8%AE%AD%E7%BB%83)

执行以下指令使用混合精度训练PP-YOLOE

```shell
python  train.py -c configs/ppyoloe_crn_l_300e_coco.yml --amp
```

**注意:** 使用默认配置训练需要设置`--amp`以避免显存溢出.

* 参数说明

| 参数            | 是否必须 | 含义                                                         |
| --------------- | -------- | ------------------------------------------------------------ |
| --config / -c   | Option   | yml配置文件路径 |
| --eval          | Option   | 在训练过程中是否开启评估(默认关闭)|
| --amp           | Option   | 启用自动混合精度训练 |
| --resume / -r   | Option   | 加载指定的预训练模型 |

* 注意：
  * 如果训练时显存out memory，将config文件中TrainReader中batch_size调小， 同时LearningRate中base_lr等比例减小。

更多yml配置详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/GETTING_STARTED.md).


### 评估

执行以下命令在单个GPU上评估COCO val2017数据集

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py -c configs/ppyoloe_crn_l_300e_coco.yml -w https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

在coco test-dev2017上评估，请先从[COCO数据集下载](https://cocodataset.org/#download)下载COCO test-dev2017数据集，然后解压到COCO数据集文件夹并像`configs/ppyolo/ppyolo_test.yml`一样配置`EvalDataset`。


### 推理测试

您可以使用以下命令进行推理测试

```shell
 python3 infer.py \
-c configs/configs/ppyoloe_crn_l_300e_coco.yml \
-w output/PPyoloe/model_final \
--infer_img=test.jpeg
```



## 推理部署
* 详情信息可以参考[连接](https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.4/configs/ppyoloe/README_cn.md#%E6%8E%A8%E7%90%86)