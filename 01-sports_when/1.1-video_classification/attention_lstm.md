<!---
简体中文 | [English](../../../en/model_zoo/recognition/attention_lstm.md)
-->
# AttentionLSTM
![image](https://user-images.githubusercontent.com/51101236/186799944-3544e72b-f4b9-45e6-b28d-2718a2cd2ae2.png)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)

## 模型简介

循环神经网络（RNN）常用于序列数据的处理，可建模视频连续多帧的时序信息，在视频分类领域为基础常用方法。
该模型采用了双向长短时记忆网络（LSTM），将视频的所有帧特征依次编码。与传统方法直接采用LSTM最后一个时刻的输出不同，该模型增加了一个Attention层，每个时刻的隐状态输出都有一个自适应权重，然后线性加权得到最终特征向量。参考论文中实现的是两层LSTM结构，而**本模型实现的是带Attention的双向LSTM**。

Attention层可参考论文[AttentionCluster](https://arxiv.org/abs/1711.09550)

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
## 模型代码

使用ResNet_50作为特征提取backbone，后端连接带Attention的双向LSTM。
不同于环境配置标准PaddleVideo代码库，具体请参考使用独立分支[Attentionlstm](https://github.com/zhangxihou/PaddleVideo_for_attentionlstm)。

## 模型训练

###  UCF101-66数据集训练

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
        name: "ResNet"
        pretrained: 将路径填写到此处
```

#### 开始训练

- ucf101-66数据集使用8卡训练，使用视频作为输入，数据的训练启动命令如下

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm  main.py  --validate -c configs/AttentionLstm_ucf101_videos.yaml
  ```

## 模型测试

命令如下：

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm  main.py  --test -c configs/AttentionLstm_ucf101_videos.yaml -w "output/AttentionLSTM/AttentionLSTM_best.pdparams"
```

当测试配置采用如下参数时，在ucf101-66的validation数据集上的测试指标如下：

avg_acc1:0.953

avg_acc5:0.989

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/AttentionLstm_ucf101_videos.yaml \
                                -p data/AttentionLSTM_ucf101.pdparams \
                                -o inference/AttentionLSTM
```

上述命令将生成预测所需的模型结构文件`AttentionLSTM.pdmodel`和模型权重文件`AttentionLSTM.pdiparams`。

各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-模型推理)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.pkl \
                           --config configs/AttentionLstm_ucf101_videos.yaml \
                           --model_file inference/AttentionLSTM/AttentionLSTM.pdmodel \
                           --params_file inference/AttentionLSTM/AttentionLSTM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```
输出示例如下：
```bash
Current video file: data/example.pkl
        top-1 class: 11
        top-1 score: 0.9841002225875854
```
可以看到，使用训练好的AttentionLSTM模型对data/example.pkl进行预测，输出的top1类别id为11，置信度为0.98。
## 参考论文

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan

