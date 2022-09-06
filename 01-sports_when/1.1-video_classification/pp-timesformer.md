<!---
[English](../../../en/model_zoo/recognition/pp-timesformer.md) | 简体中文
-->
# PP-TimeSformer视频分类模型
![image](https://user-images.githubusercontent.com/51101236/186800606-c912c4d8-4156-4d7c-ac3d-923675e285a6.png)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

我们对[TimeSformer模型](./timesformer.md)进行了改进和优化，得到了更高精度的2D实用视频分类模型**PP-TimeSformer**。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过原版，在Kinetics-400数据集上的精度如下表所示。

| Version | Top1 |
| :------ | :----: |
| Ours ([swa](#refer-anchor-1)+distill+16frame) | 79.44 |
| Ours ([swa](#refer-anchor-1)+distill)  | 78.87 |
| Ours ([swa](#refer-anchor-1)) | **78.61** |
| [mmaction2](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer#kinetics-400) | 77.92 |


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

1. 下载图像预训练模型[ViT_base_patch16_224_miil_21k.pdparams](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams)作为Backbone初始化参数，或通过wget命令下载

   ```bash
   wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
   ```

2. 打开`pptimesformer_ucf101_videos.yaml`，将下载好的权重存放路径填写到下方`pretrained:`之后

    ```yaml
    MODEL:
        framework: "RecognizerTransformer"
        backbone:
            name: "VisionTransformer"
            pretrained: 将路径填写到此处
    ```

#### 开始训练

- ucf101-66数据集使用8卡训练，训练方式的启动命令如下:

    ```bash
    # videos数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptimesformer  main.py  --validate -c configs/pptimesformer_ucf101_videos.yaml
    ```

- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

    ```bash
    export FLAGS_conv_workspace_size_limit=800 # MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1
    # videos数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptimesformer  main.py --amp --validate -c configs/pptimesformer_ucf101_videos.yaml
    ```

- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，建议配置文件的命名方式为`模型_数据集名称_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

- PP-TimeSformer模型在训练时同步进行验证，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

  ```
  Already save the best model (top1 acc)0.7258
  ```

- 由于PP-TimeSformer模型测试模式的采样方式是速度稍慢但精度高一些的**UniformCrop**，与训练过程中验证模式采用的**RandomCrop**不同，所以训练日志中记录的验证指标`topk Acc`不代表最终的测试分数，因此在训练完成之后可以用测试模式对最好的模型进行测试获取最终的指标，命令如下：

  ```bash
  # 8-frames 模型测试命令
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptimesformer  main.py  --test -c configs/pptimesformer_ucf101_videos.yaml -w "output/ppTimeSformer/ppTimeSformer_best.pdparams"

  # 16-frames模型测试命令
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptimesformer main.py --test \
  -c configs/pptimesformer_ucf101_videos.yaml \
  -o MODEL.backbone.num_seg=16 \
  -o MODEL.runtime_cfg.test.num_seg=16 \
  -o MODEL.runtime_cfg.test.avg_type='prob' \
  -o PIPELINE.test.decode.num_seg=16 \
  -o PIPELINE.test.sample.num_seg=16 \
  -w "data/ppTimeSformer_k400_16f_distill.pdparams"
  ```


  当测试配置采用如下参数时，在ucf101-66的validation数据集上的测试指标如下：
  
  avg_acc1:0.986
  
  avg_acc5:0.998


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/pptimesformer_ucf101_videos.yaml \
                                -p data/ppTimeSformer_ucf101_8f.pdparams \
                                -o inference/ppTimeSformer
```

上述命令将生成预测所需的模型结构文件`ppTimeSformer.pdmodel`和模型权重文件`ppTimeSformer.pdiparams`。

- 各参数含义可参考[模型推理方法](../../start.md#2-模型推理)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/pptimesformer_ucf101_videos.yaml \
                           --model_file inference/ppTimeSformer/ppTimeSformer.pdmodel \
                           --params_file inference/ppTimeSformer/ppTimeSformer.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9997474551200867
```

可以看到，训练好的ppTimeSformer模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为0.99。

## 参考论文

- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf), Gedas Bertasius, Heng Wang, Lorenzo Torresani
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
<div id="refer-anchor-1"></div>

- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407v3), Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov
- [ImageNet-21K Pretraining for the Masses](https://arxiv.org/pdf/2104.10972v4.pdf), Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy
