简体中文 | [English](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/picodet/README_en.md)

# 使用PP-PicoDet对图像/视频中的运动员进行检测、定位

### PP-Picodet
* `PP-PicoDet`是PaddleDetection中提出的全新轻量级系列模型，在移动端具有卓越的性能。
* 如果您想获得关于`PP-PicoDet`更详细的信息请转到[PaddleDetection/picodet](https://github.com/paddlepaddle/PaddleDetection/tree/release/2.4/configs/picodet)

![](https://ai-studio-static-online.cdn.bcebos.com/7eed45beb265486083acd6ed5d81bbd12475b7bec2864814b5d431ff8ac07ec8)

## 简介

PP-PicoDet模型有如下特点：

- 🌟 更高的mAP: 第一个在1M参数量之内`mAP(0.5:0.95)`超越**30+**(输入416像素时)。
- 🚀 更快的预测速度: 网络预测在ARM CPU下可达150FPS。
- 😊 部署友好: 支持PaddleLite/MNN/NCNN/OpenVINO等预测库，支持转出ONNX，提供了C++/Python/Android的demo。
- 😍 先进的算法: 我们在现有SOTA算法中进行了创新, 包括：ESNet, CSP-PAN, SimOTA等等。



## 基线

| 模型     | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | 参数量<br><sup>(M) | FLOPS<br><sup>(G) | 预测时延<sup><small>[CPU](#latency)</small><sup><br><sup>(ms) | 预测时延<sup><small>[Lite](#latency)</small><sup><br><sup>(ms) |  权重下载  | 配置文件 | 导出模型  |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: | :----------------------------------------: | :--------------------------------------- | :--------------------------------------- |
| PicoDet-XS |  320*320   |          23.5           |        36.1       |        0.70        |       0.67        |              3.9ms              |            7.81ms             | [model](https://paddledet.bj.bcebos.com/models/picodet_xs_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_xs_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_xs_320_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_320_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_320_coco_lcnet_non_postprocess.tar) |
| PicoDet-XS |  416*416   |          26.2           |        39.3        |        0.70        |       1.13        |              6.1ms             |            12.38ms             | [model](https://paddledet.bj.bcebos.com/models/picodet_xs_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_xs_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_xs_416_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_416_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_416_coco_lcnet_non_postprocess.tar) |
| PicoDet-S |  320*320   |          29.1           |        43.4        |        1.18       |       0.97       |             4.8ms              |            9.56ms             | [model](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_s_320_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_320_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_320_coco_lcnet_non_postprocess.tar) |
| PicoDet-S |  416*416   |          32.5           |        47.6        |        1.18        |       1.65       |              6.6ms              |            15.20ms             | [model](https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_s_416_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet_non_postprocess.tar) |
| PicoDet-M |  320*320   |          34.4           |        50.0        |        3.46        |       2.57       |             8.2ms              |            17.68ms             | [model](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_m_320_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_m_320_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_m_320_coco_lcnet_non_postprocess.tar) |
| PicoDet-M |  416*416   |          37.5           |        53.4       |        3.46        |       4.34        |              12.7ms              |            28.39ms            | [model](https://paddledet.bj.bcebos.com/models/picodet_m_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_m_416_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_m_416_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_m_416_coco_lcnet_non_postprocess.tar) |
| PicoDet-L |  320*320   |          36.1           |        52.0        |        5.80       |       4.20        |              11.5ms             |            25.21ms           | [model](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_l_320_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_320_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_320_coco_lcnet_non_postprocess.tar) |
| PicoDet-L |  416*416   |          39.4           |        55.7        |        5.80        |       7.10       |              20.7ms              |            42.23ms            | [model](https://paddledet.bj.bcebos.com/models/picodet_l_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_l_416_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_416_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_416_coco_lcnet_non_postprocess.tar) |
| PicoDet-L |  640*640   |          42.6           |        59.2        |        5.80        |       16.81        |              62.5ms              |            108.1ms          | [model](https://paddledet.bj.bcebos.com/models/picodet_l_640_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_640_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet/picodet_l_640_coco_lcnet.yml) | [w/ 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_640_coco_lcnet.tar) &#124; [w/o 后处理](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_640_coco_lcnet_non_postprocess.tar) |

<details open>
<summary><b>注意事项:</b></summary>

- <a name="latency">时延测试：</a> 我们所有的模型都在`英特尔酷睿i7 10750H`的CPU 和`骁龙865(4xA77+4xA55)`的ARM CPU上测试(4线程，FP16预测)。上面表格中标有`CPU`的是使用OpenVINO测试，标有`Lite`的是使用[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)进行测试。
- PicoDet在COCO train2017上训练，并且在COCO val2017上进行验证。使用4卡GPU训练，并且上表所有的预训练模型都是通过发布的默认配置训练得到。
- Benchmark测试：测试速度benchmark性能时，导出模型后处理不包含在网络中，需要设置`-o export.benchmark=True` 或手动修改[runtime.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/runtime.yml#L12)。

</details>

#### 其他模型的基线

| 模型     | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | 参数量<br><sup>(M) | FLOPS<br><sup>(G) | 预测时延<sup><small>[NCNN](#latency)</small><sup><br><sup>(ms) |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: |
| YOLOv3-Tiny |  416*416   |          16.6           |        33.1      |        8.86        |       5.62        |             25.42               |
| YOLOv4-Tiny |  416*416   |          21.7           |        40.2        |        6.06           |       6.96           |             23.69               |
| PP-YOLO-Tiny |  320*320       |          20.6         |        -              |   1.08             |    0.58             |    6.75                           |  
| PP-YOLO-Tiny |  416*416   |          22.7          |    -               |    1.08               |    1.02             |    10.48                          |  
| Nanodet-M |  320*320      |          20.6            |    -               |    0.95               |    0.72             |    8.71                           |  
| Nanodet-M |  416*416   |          23.5             |    -               |    0.95               |    1.2              |  13.35                          |
| Nanodet-M 1.5x |  416*416   |          26.8        |    -                  | 2.08               |    2.42             |    15.83                          |
| YOLOX-Nano     |  416*416   |          25.8          |    -               |    0.91               |    1.08             |    19.23                          |
| YOLOX-Tiny     |  416*416   |          32.8          |    -               |    5.06               |    6.45             |    32.77                          |
| YOLOv5n |  640*640       |          28.4             |    46.0            |    1.9                |    4.5              |    40.35                          |
| YOLOv5s |  640*640       |          37.2             |    56.0            |    7.2                |    16.5             |    78.05                          |

- ARM测试的benchmark脚本来自: [MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)。

## 快速开始

* 如果您想了解具体细节可以参考AIstudio-[使用Picodet/PP-yoloe检测运动员和足球](https://aistudio.baidu.com/aistudio/projectdetail/4479428?contributionType=1&sUid=206265&shared=1&ts=1661954440536)

<details open>
<summary>依赖包:</summary>

- PaddlePaddle == 2.2.2
- pycocotools == 2.0.4

</details>

<details>
<summary>安装</summary>

- [安装指导文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/INSTALL.md)
- [准备数据文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet_en.md)

</details>


<details>
<summary>配置文件与参数说明</summary>

- 相关配置位于[configs](configs)路径下
- 具体配置及模型替换等参阅：[链接](https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.4/docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

</details>

<details>
<summary>训练&评估</summary>

- 单卡GPU上训练:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python train.py -c configs/picodet_s_320_coco_lcnet.yml --eval
```
* 参数说明

| 参数            | 是否必须 | 含义                                                         |
| --------------- | -------- | ------------------------------------------------------------ |
| --config / -c   | Option   | yml配置文件路径 |
| --eval          | Option   | 在训练过程中是否开启评估(默认关闭)|
| --amp           | Option   | 启用自动混合精度训练 |
| --resume / -r   | Option   | 加载指定的预训练模型 |

* 注意：
  * 如果训练时显存out memory，将config文件中TrainReader中batch_size调小， 同时LearningRate中base_lr等比例减小。

- 评估:

```shell
 python eval.py -c configs/picodet_s_320_coco_lcnet.yml -w output/Picodet/best_model
```

详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/GETTING_STARTED.md).

</details>


<details>
<summary>推理预测</summary>

* 正确配置好configs路径下yml文件的TestDataset项(这样可以正确显示您的label)

```shell
 python3 infer.py \
-c configs/picodet_m_320_coco_lcnet.yml \
-w output/Picodet/model_final \
--infer_img=test.jpeg
```

</details>

## 部署

### 导出及转换模型
  详情请参考[Picodet-部署](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet_en.md)