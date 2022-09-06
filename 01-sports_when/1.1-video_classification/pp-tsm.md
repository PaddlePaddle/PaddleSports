<!---
[English](../../../en/model_zoo/recognition/pp-tsm.md) | 简体中文
-->
# PP-TSM视频分类模型
![image](https://user-images.githubusercontent.com/51101236/186800707-28c5e94c-4d69-43c1-b569-059c320881b2.png)

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

我们对[TSM模型](./tsm.md)进行了改进，提出了高精度2D实用视频分类模型**PP-TSM**。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过原文，在Kinetics-400数据集上的精度如下表所示。模型优化解析请参考[**PP-TSM模型精度优化Tricks详解**](https://zhuanlan.zhihu.com/p/382134297)。

| Version | Sampling method | Top1 |
| :------ | :----------: | :----: |
| Ours (distill) | Dense | **76.16** |
| Ours | Dense | 75.69 |
| [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md) | Dense | 74.55 |
| [mit-han-lab](https://github.com/mit-han-lab/temporal-shift-module) | Dense | 74.1 |

| Version | Sampling method | Top1 |
| :------ | :----------: | :----: |
| Ours (distill) | Uniform | **75.11** |
| Ours | Uniform | 74.54 |
| [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md) |  Uniform | 71.90 |
| [mit-han-lab](https://github.com/mit-han-lab/temporal-shift-module)  | Uniform | 71.16 |


## 数据准备

UCF101数据下载及准备请参考[UCF-101数据准备](../../dataset/ucf101.md)
UCF101数据中抽取66类体育类别识别数据，体育数据类别为
```bash
sports_type=['BalanceBeam',\
    'BaseballPitch',\
    'Basketball',\
    'BasketballDunk',\
    'BenchPress',\
    'Biking',\
    'Billiards',\
    'BodyWeightSquats',\
    'Bowling',\
    'BoxingPunchingBag',\
    'BoxingSpeedBag',\
    'BreastStroke',\
    'CleanAndJerk',\
    'CliffDiving',\
    'CricketBowling',\
    'CricketShot',\
    'Diving',\
    'Fencing',\
    'FieldHockeyPenalty',\
    'FloorGymnastics',\
    'FrisbeeCatch',\
    'FrontCrawl',\
    'GolfSwing',\
    'Hammering',\
    'HammerThrow',\
    'HandstandPushups',\
    'HandstandWalking',\
    'HighJump',\
    'HorseRiding',\
    'HulaHoop',\
    'IceDancing',\
    'JavelinThrow',\
    'JugglingBalls',\
    'JumpingJack',\
    'JumpRope',\
    'Kayaking',\
    'LongJump',\
    'Lunges',\
    'Nunchucks',\
    'ParallelBars',\
    'PoleVault',\
    'PullUps',\
    'PushUps',\
    'Rafting',\
    'RockClimbingIndoor',\
    'RopeClimbing',\
    'Rowing',\
    'Shotput',\
    'SkateBoarding',\
    'Skiing',\
    'SkyDiving',\
    'SoccerJuggling',\
    'SoccerPenalty',\
    'StillRings',\
    'SumoWrestling',\
    'Swing',\
    'TableTennisShot',\
    'TaiChi',\
    'TennisSwing',\
    'ThrowDiscus',\
    'TrampolineJumping',\
    'UnevenBars',\
    'VolleyballSpiking',\
    'WallPushups',\
    'YoYo']
```
## 模型训练

### UCF101-66数据集训练

#### 下载并添加预训练模型

下载图像蒸馏预训练模型[ResNet50_vd_ssld_v2.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams)作为Backbone初始化参数，或是通过命令行下载

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

并将文件路径添加到配置文件中的`MODEL.framework.backbone.pretrained`字段，如下：

```yaml
MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "ResNetTweaksTSM"
        pretrained: 将路径填写到此处
```

- 如果使用ResNet101作为Backbone进行训练，请下载预训练模型[ResNet101_vd_ssld_pretrained.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ResNet101_vd_ssld_pretrained.pdparams).

#### 开始训练

- UCF101-66数据集使用8卡训练，frames格式数据，uniform训练方式的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_ucf101-66_frames_uniform.yaml
```

- UCF101-66数据集使用8卡训练，videos格式数据，uniform训练方式的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_ucf101-66_videos_uniform.yaml
```

- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

```bash
export FLAGS_conv_workspace_size_limit=800 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --amp --validate -c configs/recognition/pptsm/pptsm_ucf101-66_frames_uniform.yaml
```

- UCF101-66数据集frames格式数据，dense训练方式的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_ucf101-66_frames_dense.yaml
```

- UCF101-66数据集frames格式数据，dense训练方式，ResNet101作为Backbone的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_ucf101-66_frames_dense_r101.yaml
```

- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，配置文件命名方式为`模型_数据集_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

- 对Uniform采样方式，PP-TSM模型在训练时同步进行测试，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

```txt
Already save the best model (top1 acc)0.7454
```

- 对dense采样方式，需单独运行测试代码，其启动命令如下：

```bash
python3 main.py --test -c configs/recognition/pptsm/pptsm_ucf101-66_frames_dense.yaml -w output/ppTSM/ppTSM_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。


UCF101-66数据集测试精度:
```bash
avg_acc1:0.974
avg_acc5:0.994
```

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/pptsm/pptsm_ucf101-66_frames_uniform.yaml \
                                -p data/ppTSM_ucf101-66_uniform.pdparams \
                                -o inference/ppTSM
```

上述命令将生成预测所需的模型结构文件`ppTSM.pdmodel`和模型权重文件`ppTSM.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/pptsm/pptsm_ucf101-66_frames_uniform.yaml \
                           --model_file inference/ppTSM/ppTSM.pdmodel \
                           --params_file inference/ppTSM/ppTSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```


## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
