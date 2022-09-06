# SoccerNet_ReID_PaddleClas

## 1. 介绍
* 一个基于 PaddleClas 套件和 [SoccerNet ReID](https://github.com/SoccerNet/sn-reid) 数据集开发的足球运动员重识别（ReID）的基线

* 支持 ResNet、MobileNet 等多种经典 Backbone，模型训练、评估、推理和部署全流程支持


## 2. 演示
* ReID 即通过一个输入的 Query 图像，然后从图库（Gallery）寻找与之最相似的 N 个图像：

    ![](https://ai-studio-static-online.cdn.bcebos.com/2d21755eef444512af34d646c5fc45481a006973c16047d4b09e6944b11cf0b9)
    

## 3. 目录
* 本项目目录结构如下：

    * configs -> 模型配置文件

    * tools -> 训练和评估脚本
    
    
    * PaddleClas -> PaddleClas 套件源码

## 4. 数据
### 4.1 数据简介
* SoccerNet Re-Identification （ReID） 数据集由 340.993 名球员缩略图组成，这些缩略图是从 6 个主要联赛的 400 场足球比赛的广播视频的图像帧中提取的。
* 该挑战的目标是在多个摄像机视点中重新识别足球运动员，这些视点描绘了比赛中的相同动作。

* 我们将广播视频中已看到动作的第一帧称为“动作帧”。

* 描绘相同动作（在大致相同的时间戳）并在广播视频中稍后出现的其他帧称为“重播帧”。

* 挑战的目标是在不同的重播帧中重新识别动作帧中的每个足球运动员。

* 为此，给定操作帧中的每个边界框将被视为查询，来自相应重播帧的边界框将根据其到每个查询的距离进行排名。

* SoccerNet ReID数据集分为训练，验证，测试和挑战集。

* 挑战集的标签是保密的，因为挑战的参与者将根据他们在挑战上的表现进行排名。

* 挑战赛的获胜者将是挑战赛中 mAP 得分最高的参赛者。

* 与传统的街道监控类型再识别数据集相比，SoccerNet-v3 ReID数据集特别具有挑战性，因为来自同一支球队的足球运动员具有非常相似的外表，这使得很难区分他们。

* 另一方面，每个标识都有一些样本图像分辨率有很大的差异，这使得模型更难训练。

    > 重要说明：  
    玩家身份标签派生自操作中边界框之间的链接，仅在给定操作中有效。  
    所以玩家身份标签在操作中不成立，并且给定的玩家对于他被发现的每个操作都有不同的身份。  
    因此，在评估过程中，只有同一操作中的样本相互匹配。  
    在训练阶段，参与者需要实施策略，以减轻来自不同动作的图像样本的负面影响，这些动作描绘了同一名球员，但用不同的身份标记。


### 4.2 数据格式
* ReID 数据集以缩略图形式储存，样例如下：

    ![](https://ai-studio-static-online.cdn.bcebos.com/16c1493505ba46089273d90691b000acfddba42e4c1e46779086f5c827a84a01)
    ![](https://ai-studio-static-online.cdn.bcebos.com/3021f6b496b844968bd289f6cd37fd7294681302ecf447129c116bcc036a5e40)
    ![](https://ai-studio-static-online.cdn.bcebos.com/9adc00f3516b415ea96bb91f77dbfcbfb3c984c6b2f94bc28b21532a340b7ab2)
    ![](https://ai-studio-static-online.cdn.bcebos.com/1648998a757d4fdfb6141993feedd27cfd94bb91f2e44ce6ac484005bdbc6810)
    ![](https://ai-studio-static-online.cdn.bcebos.com/21f215d427ca4e42a405fe323561ceb0bf1a8d620c9a4f84a0df96db5fefc588)
    
    

* 真实值和检测结果存储在逗号分隔的 txt 文件中，共 10 列，样例如下：

    ```json
    {
        "0": {
            "bbox_idx": 0,
            "action_idx": 0,
            "person_uid": 0,
            "frame_idx": 0,
            "clazz": "Player_team_left",
            "id": "None",
            "UAI": "000r000_004597cb000d",
            "relative_path": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/0",
            "height": 74,
            "width": 32
        },
        "1": {
            "bbox_idx": 1,
            "action_idx": 0,
            "person_uid": 1,
            "frame_idx": 0,
            "clazz": "Player_team_left",
            "id": "None",
            "UAI": "000r000_004597cb000e",
            "relative_path": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/0",
            "height": 51,
            "width": 20
        },
        "2": {
            "bbox_idx": 2,
            "action_idx": 0,
            "person_uid": 2,
            "frame_idx": 0,
            "clazz": "Player_team_left",
            "id": "b",
            "UAI": "000r000_004597cb000f",
            "relative_path": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/0",
            "height": 63,
            "width": 24
        },
        "3": {
            "bbox_idx": 3,
            "action_idx": 0,
            "person_uid": 3,
            "frame_idx": 0,
            "clazz": "Player_team_left",
            "id": "2",
            "UAI": "000r000_004597cb0010",
            "relative_path": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/0",
            "height": 75,
            "width": 31
        },
        "4": {
            "bbox_idx": 4,
            "action_idx": 0,
            "person_uid": 4,
            "frame_idx": 0,
            "clazz": "Player_team_left",
            "id": "g",
            "UAI": "000r000_004597cb0011",
            "relative_path": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/0",
            "height": 64,
            "width": 41
        },
        "5": {
            "bbox_idx": 5,
            "action_idx": 0,
            "person_uid": 5,
            "frame_idx": 0,
            "clazz": "Player_team_left",
            "id": "f",
            "UAI": "000r000_004597cb0012",
            "relative_path": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/0",
            "height": 67,
            "width": 20
        }
        ...
    }
    ```

* 数据集符合 MOT20 格式要求，由于某些值未使用，所以被标注为 -1

### 4.3 数据结构
* Soccernet-v3 ReID 数据集被组织在如下的树文件夹结构中：

    `root` -> `{train, valid, test, challenge}` -> `championship` -> `season` -> `game` -> `action` -> `image files`

## 6. 使用

### 6.1 克隆源码


```bash
$ git clone https://gitee.com/PaddlePaddle/PaddleClas

$ cp -r configs PaddleClas/ppcls
$ cp -r tools PaddleClas
```

### 6.2 安装依赖


```bash
$ cd PaddleClas

$ pip install -r requirements.txt
```

### 6.3 数据解压
* 解压数据集至 PaddleClas/dataset/SNReID 目录

```yaml
- SNReID
  - train
    ...
  - valid
    ...
  - test
    ...
  - challenge
    ...
```

### 6.4 模型训练
* 指定配置文件进行模型训练


```bash
$ python tools/train_snreid.py -c ppcls/configs/snreid/baseline.yaml
```

### 6.5 模型验证
* 根据官方评估脚本进行修改

* 使用如下命令实现模型精度评估：




```bash
$ python tools/eval_snreid.py -c ppcls/configs/snreid/baseline.yaml -o Global.pretrained_model=./output/RecModel/latest
```

    [2022/08/10 00:17:01] ppcls INFO: [Eval][Epoch 0][Avg]mAP: 0.57672, rank-1: 0.45455


### 6.6 模型推理
#### 6.6.1 导出推理模型
* 模型推理之前需要导出推理模型

* 使用如下命令进行模型导出：


```bash
$ python tools/export_model.py -c ppcls/configs/snreid/baseline.yaml -o Global.pretrained_model=./output/RecModel/latest
```

#### 6.6.2 获取特征向量
* 通过导出的推理模型，就可以输入照片然后得到这张图片所对应的特征向量

* 这个特征向量可以在高纬度的向量空间中描述不同图片的相似程度

* 然后通过一些向量检索的方法就可以实现 ReID 功能了


```bash
$ cd deploy

$ python python/predict_rec.py \
    -c ../ppcls/configs/snreid/inference.yaml \
    -o Global.rec_inference_model_dir="../inference"
```

### 6.7 更多详情
* 模型部署和更多使用指南请参考 PaddleClas 官方文档

## 7. 参考
* 相关链接：

    * [SoccerNet: Soccer Video Understanding Benchmark Suite](https://www.soccer-net.org/home)

    * [SoccerNet ReID Official Website](https://www.soccer-net.org/tasks/re-identification)
    
    * [SoccerNet ReID Development Kit](https://github.com/SoccerNet/sn-reid)
    
    * [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
    
    
