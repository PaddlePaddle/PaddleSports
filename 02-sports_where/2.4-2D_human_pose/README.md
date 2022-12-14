# 关键点检测系列模型

<div align="center">
  <img src="./football_keypoint.gif" width='800'/>
</div>

## 目录
- [简介](#简介)
- [模型推荐](#模型推荐)
- [模型库](#模型库)
- [快速开始](#快速开始)
  - [环境安装](#1环境安装)
  - [模型训练](#2模型训练)
  - [模型评估](#3模型评估)
  - [模型推理](#4模型推理)
  - [完整部署教程及Demo](#5完整部署教程及Demo)
- [BenchMark](#benchmark)

## 简介

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 关键点检测能力紧跟业内最新最优算法方案，包含Top-Down、Bottom-Up两套方案，Top-Down先检测主体，再检测局部关键点，优点是精度较高，缺点是速度会随着检测对象的个数增加，Bottom-Up先检测关键点再组合到对应的部位上，优点是速度快，与检测对象个数无关，缺点是精度较低。

同时，PaddleDetection提供针对移动端设备优化的自研实时关键点检测模型[PP-TinyPose](./tiny_pose_readme.md)，以满足用户的不同需求。

## 模型推荐
### 移动端模型推荐

| 检测模型                                                     | 关键点模型                            |             输入尺寸             |         COCO数据集精度          |          平均推理耗时 (FP16)           | 参数量 （M）                |          Flops (G)          |                           模型权重                           |                  Paddle-Lite部署模型（FP16)                  |
| :----------------------------------------------------------- | :------------------------------------ | :------------------------------: | :-----------------------------: | :------------------------------------: | --------------------------- | :-------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [PicoDet-S-Pedestrian](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/picodet/legacy_model/application/pedestrian_detection/picodet_s_192_pedestrian.yml) | [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/tiny_pose/tinypose_128x96.yml)  | 检测：192x192<br>关键点：128x96  | 检测mAP：29.0<br>关键点AP：58.1 | 检测耗时：2.37ms<br>关键点耗时：3.27ms | 检测：1.18<br/>关键点：1.36 | 检测：0.35<br/>关键点：0.08 | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian.pdparams)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams) | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian_fp16.nb)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_fp16.nb) |
| [PicoDet-S-Pedestrian](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/picodet/legacy_model/application/pedestrian_detection/picodet_s_320_pedestrian.yml) | [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/tiny_pose/tinypose_256x192.yml) | 检测：320x320<br>关键点：256x192 | 检测mAP：38.5<br>关键点AP：68.8 | 检测耗时：6.30ms<br>关键点耗时：8.33ms | 检测：1.18<br/>关键点：1.36 | 检测：0.97<br/>关键点：0.32 | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.pdparams)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian_fp16.nb)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_fp16.nb) |


*详细关于PP-TinyPose的使用请参考[文档](./tiny_pose_readme.md)。

### 服务端模型推荐

| 检测模型                                                     | 关键点模型                                 |             输入尺寸             |         COCO数据集精度          |       参数量 （M）       |        Flops (G)         |                           模型权重                           |
| :----------------------------------------------------------- | :----------------------------------------- | :------------------------------: | :-----------------------------: | :----------------------: | :----------------------: | :----------------------------------------------------------: |
| [PP-YOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [HRNet-w32](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/hrnet/hrnet_w32_384x288.yml) | 检测：640x640<br>关键点：384x288 | 检测mAP：49.5<br>关键点AP：77.8 | 检测：54.6<br/>关键点：28.6 | 检测：115.8<br/>关键点：17.3 | [检测](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams)<br>[关键点](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) |
| [PP-YOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [HRNet-w32](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/hrnet/hrnet_w32_256x192.yml) | 检测：640x640<br>关键点：256x192 | 检测mAP：49.5<br>关键点AP：76.9 | 检测：54.6<br/>关键点：28.6 | 检测：115.8<br/>关键点：7.68 | [检测](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams)<br>[关键点](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams) |


## 模型库

COCO数据集
| 模型              |  方案              |输入尺寸 | AP(coco val) |                           模型下载                           | 配置文件 |  
| :---------------- | -------- | :----------: | :----------------------------------------------------------: | ----------------------------------------------------| ------- |
| HigherHRNet-w32       |Bottom-Up| 512      |     67.1     | [higherhrnet_hrnet_w32_512.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml)       |
| HigherHRNet-w32       | Bottom-Up| 640      |     68.3     | [higherhrnet_hrnet_w32_640.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_640.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_640.yml)       |
| HigherHRNet-w32+SWAHR |Bottom-Up|  512      |     68.9     | [higherhrnet_hrnet_w32_512_swahr.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512_swahr.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512_swahr.yml) |
| HRNet-w32             | Top-Down| 256x192  |     76.9     | [hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/hrnet/hrnet_w32_256x192.yml)                    |
| HRNet-w32             |Top-Down| 384x288  |     77.8     | [hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/hrnet/hrnet_w32_384x288.yml)                     |
| HRNet-w32+DarkPose             |Top-Down| 256x192  |     78.0     | [dark_hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/hrnet/dark_hrnet_w32_256x192.yml)                 |
| HRNet-w32+DarkPose             |Top-Down| 384x288  |     78.3     | [dark_hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/hrnet/dark_hrnet_w32_384x288.yml)                   |
| WiderNaiveHRNet-18         | Top-Down|256x192  |     67.6(+DARK 68.4)     | [wider_naive_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/wider_naive_hrnet_18_256x192_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/lite_hrnet/wider_naive_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   |Top-Down| 256x192  |     66.5     | [lite_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_256x192_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/lite_hrnet/lite_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   |Top-Down| 384x288  |     69.7     | [lite_hrnet_18_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_384x288_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/lite_hrnet/lite_hrnet_18_384x288_coco.yml)     |
| LiteHRNet-30                   | Top-Down|256x192  |     69.4     | [lite_hrnet_30_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_256x192_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/lite_hrnet/lite_hrnet_30_256x192_coco.yml)     |
| LiteHRNet-30                   |Top-Down| 384x288  |     72.5     | [lite_hrnet_30_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_384x288_coco.pdparams) | [config]([./lite_hrnet/lite_hrnet_30_384x288_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/lite_hrnet/lite_hrnet_30_384x288_coco.yml))     |


备注： Top-Down模型测试AP结果基于GroundTruth标注框

MPII数据集
| 模型  | 方案| 输入尺寸 | PCKh(Mean) | PCKh(Mean@0.1) |                           模型下载                           | 配置文件                                     |
| :---- | ---|----- | :--------: | :------------: | :----------------------------------------------------------: | -------------------------------------------- |
| HRNet-w32 | Top-Down|256x256  |    90.6    |      38.5      | [hrnet_w32_256x256_mpii.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x256_mpii.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml) |

场景模型
| 模型 | 方案 | 输入尺寸 | 精度 | 预测速度 |模型权重 | 部署模型 | 说明|
| :---- | ---|----- | :--------: | :--------: | :------------: |:------------: |:-------------------: |
| HRNet-w32 + DarkPose | Top-Down|256x192  |  AP: 87.1 (业务数据集)| 单人2.9ms |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.pdparams) |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) | 针对摔倒场景特别优化，该模型应用于[PP-Human](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/pipeline/README.md) |

我们同时推出了基于LiteHRNet（Top-Down）针对移动端设备优化的实时关键点检测模型[PP-TinyPose](./tiny_pose_readme.md), 欢迎体验。

## 快速开始

### 1、环境安装

```shell
pip install -r requirements.txt
``` 


### 2、模型训练

PP-TinyPose模型训练：

```shell
python tools/train.py --config configs/tinypose_128x96.yml --eval
``` 

### 3、模型评估

```shell
python tools/eval.py --config configs/tinypose_128x96.yml
```

### 4、模型推理

```shell
python tools/infer.py --config configs/tinypose_128x96.yml \
                       --infer_img demo.jpg
```

--infer_img后面为推理图像的路径。

### 5、完整部署教程及Demo

​ PaddleDetection提供了PaddleInference(服务器端)、PaddleLite(移动端)、第三方部署(MNN、OpenVino)支持。无需依赖训练代码，deploy文件夹下相应文件夹提供独立完整部署代码。 详见 [部署文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/README.md)介绍。

## BenchMark

我们给出了不同运行环境下的测试结果，供您在选用模型时参考。详细数据请见[Keypoint Inference Benchmark](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/keypoint/KeypointBenchmark.md)。

## 引用
```
@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={TPAMI},
  year={2019}
}

@InProceedings{Zhang_2020_CVPR,
    author = {Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
    title = {Distribution-Aware Coordinate Representation for Human Pose Estimation},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```

