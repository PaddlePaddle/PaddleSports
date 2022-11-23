简体中文 | [English](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/faster_rcnn/README.md)
# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Model Zoo

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Faster         |    1    |   1x    |     ----     |  36.7  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_1x_coco.yml) |
| ResNet50-vd          | Faster         |    1    |   1x    |     ----     |  37.6  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_1x_coco.yml) |
| ResNet101            | Faster         |    1    |   1x    |     ----     |  39.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_1x_coco.pdparams) | [配置文件](./faster_rcnn_r101_1x_coco.yml) |
| ResNet34-FPN         | Faster         |    1    |   1x    |     ----     |  37.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r34_fpn_1x_coco.yml) |
| ResNet34-FPN-MultiScaleTest | Faster  |    1    |   1x    |     ----     |  38.2  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_fpn_multiscaletest_1x_coco.pdparams) | [配置文件](./faster_rcnn_r34_fpn_multiscaletest_1x_coco.yml) |
| ResNet34-vd-FPN      | Faster         |    1    |   1x    |     ----     |  38.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_vd_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r34_vd_fpn_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   1x    |     ----     |  38.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   2x    |     ----     |  40.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r50_fpn_2x_coco.yml) |
| ResNet50-vd-FPN      | Faster         |    1    |   1x    |     ----     |  39.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_1x_coco.yml) |
| ResNet50-vd-FPN      | Faster         |    1    |   2x    |     ----     |  40.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_2x_coco.yml) |
| ResNet101-FPN        | Faster         |    1    |   2x    |     ----     |  41.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r101_fpn_2x_coco.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   1x    |     ----     |  42.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r101_vd_fpn_1x_coco.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   2x    |     ----     |  43.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r101_vd_fpn_2x_coco.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   1x    |     ----     |  43.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_x101_vd_64x4d_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_x101_vd_64x4d_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   2x    |     ----     |  44.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_x101_vd_64x4d_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_x101_vd_64x4d_fpn_2x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Faster       |    1    |   1x    |     ----     |  41.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_ssld_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_ssld_1x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Faster       |    1    |   2x    |     ----     |  42.3  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_ssld_2x_coco.yml) |
| Swin-Tiny-FPN | Faster       |    2    |   1x    |     ----     |  42.6  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_swin_tiny_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_swin_tiny_fpn_1x_coco.yml) |
| Swin-Tiny-FPN | Faster       |    2    |   2x    |     ----     |  44.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_swin_tiny_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_swin_tiny_fpn_2x_coco.yml) |
| Swin-Tiny-FPN | Faster       |    2    |   3x    |     ----     |  45.3  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_swin_tiny_fpn_3x_coco.pdparams) | [配置文件](./faster_rcnn_swin_tiny_fpn_3x_coco.yml) |

## 使用说明

[comment]: <> (* 如果您想了解具体细节可以参考AIstudio-[使用Picodet/PP-yoloe检测运动员和足球]&#40;https://aistudio.baidu.com/aistudio/projectdetail/4479428?contributionType=1&sUid=206265&shared=1&ts=1661954440536&#41;)


### 训练

* 多卡训练：详情信息可以参考[连接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/faster_rcnn/README.md)

执行以下指令使用混合精度训练

```shell
python  train.py -c configs/faster_rcnn_r50_1x_coco.yml --amp
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
CUDA_VISIBLE_DEVICES=0 python eval.py -c configs/faster_rcnn_r50_1x_coco.yml -w your weights
```

在coco test-dev2017上评估，请先从[COCO数据集下载](https://cocodataset.org/#download)下载COCO test-dev2017数据集，然后解压到COCO数据集文件夹并像`configs/ppyolo/ppyolo_test.yml`一样配置`EvalDataset`。


### 推理测试

您可以使用以下命令进行推理测试

```shell
 python3 infer.py \
-c configs/configs/configs/faster_rcnn_r50_1x_coco.yml \
-w output/mask_rcnn/model_final \
--infer_img=test.jpeg
```