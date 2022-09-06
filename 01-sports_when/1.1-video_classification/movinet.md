<!---
[English](../../../en/model_zoo/recognition/movinet.md) | 简体中文
-->
# MoViNet视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

MoViNet是Google Research研发的移动视频网络。它使用神经结构搜索的方法来搜索MoViNet空间结构，使用因果卷积算子和流缓冲区来弥补准确率的损失，Temporal Ensembles提升准确率，是一个可以用于在线推理视频流的，轻量高效视频模型。

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

UCF101数据集中视频帧数差别较大，训练及测试时视频帧数不得小于设置的num_seg，因此生成train_frames.list后需要用脚本筛选帧数过低的视频，可参考一下脚本：

```bash
train_list_long=open('train_frames_long.list','w')

for line in open('train_frames.list'):
    frs = int(line.split(' ')[1])
    if frs >= 50:
        train_list_long.write(line)

train_list_long.close()
```

## 模型训练

### UCF101-66数据集训练

#### 开始训练

- UCF101-66数据集使用8卡训练，frames格式数据，uniform训练方式的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_movinet  main.py  --validate -c configs/recognition/movinet/movinet_ucf101-66_frames.yaml
```

- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，配置文件命名方式为`模型_数据集_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../tutorials/config.md)。

## 模型测试

- MoViNet模型在训练时同步进行测试，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

```txt
Already save the best model (top1 acc)0.6489
```

- 若需单独运行测试代码，其启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/movinet/movinet_ucf101-66_frame.yaml -w output/MoViNet/MoViNet_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

UCF101-66数据集测试精度:
```bash
avg_acc1:0.979
avg_acc5:1.000
```

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/movinet/movinet_ucf101-66_frame.yaml \
                                -p data/MoViNetA0_ucf101-66.pdparams \
                                -o inference/MoViNetA0
```

上述命令将生成预测所需的模型结构文件`MoViNetA0.pdmodel`和模型权重文件`MoViNetA0.pdiparams`。

各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/movinet/movinet_ucf101-66_frame.yaml \
                           --model_file inference/MoViNetA0/MoViNet.pdmodel \
                           --params_file inference/MoViNetA0/MoViNet.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:
```txt
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.7667049765586853
```

## 参考论文

- [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511)
