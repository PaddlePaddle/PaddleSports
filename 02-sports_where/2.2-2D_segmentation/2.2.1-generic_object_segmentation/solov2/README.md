简体中文 | [English](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/solov2/README.md)

# SOLOv2 for instance segmentation

## Introduction

SOLOv2 (Segmenting Objects by Locations) is a fast instance segmentation framework with strong performance. We reproduced the model of the paper, and improved and optimized the accuracy and speed of the SOLOv2.

**Highlights:**

- Training Time: The training time of the model of `solov2_r50_fpn_1x` on Tesla v100 with 8 GPU is only 10 hours.

## Model Zoo

| Detector  | Backbone                | Multi-scale training  | Lr schd |  Mask AP<sup>val</sup> |  V100 FP32(FPS) |    GPU  |    Download                  | Configs |
| :-------: | :---------------------: | :-------------------: | :-----: | :--------------------: | :-------------: | :-----: | :---------: | :------------------------: |
| YOLACT++  |  R50-FPN    | False      |  80w iter     |   34.1 (test-dev) |  33.5  | Xp |  -  |  -   |
| CenterMask | R50-FPN | True        |   2x    |     36.4        |  13.9  | Xp |   -  |  -  |
| CenterMask | V2-99-FPN | True        |   3x    |     40.2       |  8.9  | Xp |   -  |  -  |
| PolarMask | R50-FPN | True        |   2x    |     30.5        |  9.4  | V100 |   -  |  -  |
| BlendMask | R50-FPN | True        |   3x    |     37.8        |  13.5  | V100 |   -  |  -  |
| SOLOv2 (Paper) | R50-FPN | False        |   1x    |     34.8        |  18.5  | V100 |   -  |  -  |
| SOLOv2 (Paper) | X101-DCN-FPN | True        |   3x    |     42.4        |  5.9  | V100 |   -  |  -  |
| SOLOv2 | R50-FPN                 |  False                |   1x    |    35.5         |  21.9     | V100 |  [model](https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/solov2/solov2_r50_fpn_1x_coco.yml) |
| SOLOv2 | R50-FPN                 |  True                |   3x    |     38.0         |   21.9    | V100 |  [model](https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_3x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/solov2/solov2_r50_fpn_3x_coco.yml) |
| SOLOv2 | R101vd-FPN                 |  True                |   3x    |     42.7         |   12.1    | V100 |  [model](https://paddledet.bj.bcebos.com/models/solov2_r101_vd_fpn_3x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/solov2/solov2_r101_vd_fpn_3x_coco.yml) |

**Notes:**

- SOLOv2 is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.

## Enhanced model
| Backbone                | Input size  | Lr schd | V100 FP32(FPS) | Mask AP<sup>val</sup> |         Download                  | Configs |
| :---------------------: | :-------------------: | :-----: | :------------: | :-----: | :---------: | :------------------------: |
| Light-R50-VD-DCN-FPN          |  512     |   3x    |     38.6          |  39.0   | [model](https://paddledet.bj.bcebos.com/models/solov2_r50_enhance_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/solov2/solov2_r50_enhance_coco.yml) |

**Optimizing method of enhanced model:**
- Better backbone network: ResNet50vd-DCN
- A better pre-training model for knowledge distillation
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- Synchronized Batch Normalization
- Multi-scale training
- More data augmentation methods
- DropBlock

## 使用说明

[comment]: <> (* 如果您想了解具体细节可以参考AIstudio-[使用Picodet/PP-yoloe检测运动员和足球]&#40;https://aistudio.baidu.com/aistudio/projectdetail/4479428?contributionType=1&sUid=206265&shared=1&ts=1661954440536&#41;)


### 训练

* 多卡训练：详情信息可以参考[连接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/solov2/README.md)

执行以下指令使用混合精度训练

```shell
python  train.py -c configs/solov2_r50_enhance_coco.yml --amp
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
CUDA_VISIBLE_DEVICES=0 python eval.py -c configs/solov2_r50_enhance_coco.yml -w your_weights
```

在coco test-dev2017上评估，请先从[COCO数据集下载](https://cocodataset.org/#download)下载COCO test-dev2017数据集，然后解压到COCO数据集文件夹并配置`EvalDataset`。


### 推理测试

您可以使用以下命令进行推理测试

```shell
 python3 infer.py \
-c configs/solov2_r50_enhance_coco.yml \
-w output/Solov2/model_final \
--infer_img=test.jpeg
```
