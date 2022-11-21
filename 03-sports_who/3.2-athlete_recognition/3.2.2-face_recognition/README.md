## Face.Eval运动员主体人脸识别

## 简介

通过这个repo，您可以使用自己的数据集构建训练自己的人脸检测模型。
其中包含了人脸数据处理、训练、以及部署

该库可以帮助研究人员/工程师快速开发深度人脸识别模型和算法，方便实际使用和开发部署

* 运动员运动过程中识别效果

[![img](https://ai-studio-static-online.cdn.bcebos.com/c231a51ac63a4f708371fa43db98787fb3637d64f4a04c4b88ee381f438f4738)](https://player.bilibili.com/player.html?aid=941024504&bvid=BV11W4y1m7Vc&cid=776635613&page=1)

## 快速开始

#### AIstudio 示例项目连接[Face.eval人脸识别](https://aistudio.baidu.com/aistudio/projectdetail/2305787?contributionType=1&sUid=206265&shared=1&ts=1661685508709)

#### 环境准备
```shell
paddlepaddle-gpu==2.1.2
paddleslim==2.1.0
```

#### 数据集准备

* 准备您的训练/验证/测试数据集（数据集可以参考Data zoo）
,在使用数据集之前确保您的数据集按照以下结构存放。
  ```
  ./data/db_name/
        -> id1/
            -> 1.jpg
            -> ...
        -> id2/
            -> 1.jpg
            -> ...
        -> ...
            -> ...
            -> ...
  ```
  数据的读取具体方式参考 `tools/dataload.py`

#### FaceAlign
  * 人脸检测和对齐基于MTCNN
  * 文件夹 `./align` 下包含了人脸检测人脸数据对齐等人脸数据处理功能

  * 您可以将指定您的文件夹对数据进行清洗

    ```
    python analysis_clear.py -source_root='../data/lfw' -cpu_num=4
    ```
    
  * 处理数据，生成可训练的数据

    ```
    python face_align.py -source_root='../data/lfw' -dest_root='../data/output_align'
    ```

  *  移除低频小样本类（类别样本数少于min_num的）remove low-shot

    ```
    python remove_low-shot.py -root '../data/lfw_align' -min_num 3
    ```

  * 划分数据集
    ```
    python move_data.py -source_root='csapeal_align' -samples_num=10
    ```

#### Train

  * 准备训练
    
    配置并且优化您的config
    ```
        SEED = 1337, # 随机种子
        DATA_ROOT = 'data/lfw_align', # 训练数据目录
        BATCH_SAMPLE_NUM=64000, # 每个epoch均衡采样采集的样本数量，
        SNAPSHOT=10, # 每个epoch输出的log次数
        WITH_VAIDL = True, # 采用验证数据集验证
        VAL_DATA_ROOT = 'data/casia500', # 验证数据集目录
    
        MODEL_ROOT = 'output', # 模型 checkpoint 保存路径
        BACKBONE_RESUME_ROOT = 'output/', # 需要加载的checkpoint backbone 模型路径
        HEAD_RESUME_ROOT = 'output/', # 需要加载的checkpoint head 模型路径
        PRETRAINED_MODEL = True, # 加载一个预训练模型
    
        # 配置模型，提供了多种Backbone、Head、Loss
        BACKBONE_NAME = 'ppResNet_50', # support: ['ppResNet_50','ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152','GhostNet']
        HEAD_NAME = 'ArcFace', # support:  ['ArcFace', 'CosFace', 'SphereFace', 'Am_softmax','Softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']
        
        INPUT_SIZE = [112, 112], # support: [112, 224]
        RGB_MEAN=[127.5, 127.5, 127.5],  
        RGB_STD=[127.5, 127.5, 127.5],
    
        EMBEDDING_SIZE = 128, # feature dimension
        BATCH_SIZE = 512, # train 224x224 use 27167MiB bs=256l
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 1e-2, # initial LR
        NUM_EPOCH = 50, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [20, 30, 45], # epoch stages to decay learning rate
        GPU_ID = [0], # specify your GPU ids , Mult_gpu training please set GPUs in file start_mult_gpu_train.py
        PIN_MEMORY = True,
        NUM_WORKERS = 4,

        SAVE_CHECKPOINT = False,# save checkpoint 
        SAVE_QUANT_MODEL = False, # 是否使用量化训练
    ```

### Data-Zoo

|数据集              |Version      |Identity |Image      |Frame|Video|下载地址|
|:---:              |:----:       |:-----:  |:-----:    |:-----:|:-----:|:-----:|
|[LFW]()            |Align        |454      |4,877      |-    |-|[Google Drive](), [Baidu Drive](https://pan.baidu.com/s/1VzSI_xqiBw-uHKyRbi6zzw)
|[CASIA-WebFace]()  |Align_112x112|10,575   |455,594    |-    |-|[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/103163), [-]()
|[VggFace2]()       |Raw          |8,631    |3,141,890  |-    |-|[-](), [-]()
|[Vgg-Face2-clean]()|Align_112x112|1,333    |440,028    |-    |-|[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/102305), [-]()
|[Vgg-Face2-part]() |Align_112x112|1,608    |391,967    |-    |-|[AI Studio](), [-]()
|[VggFace2-sub]()   |Align_112x112|734      |184,089    |-    |-|[AI Studio](), [-]()
|[VggFace2-tiny]()  |Align_112x112|98       |6,522      |-    |-|[AI Studio](), [-]()
|[CSA-PEAL]()       |Align_112x112|1,040    |23,829     |-    |-|[-](), [-]()

### Model-Zoo

|benchmark          |BACKBONE       |HEAD         |embedding size   |Top1准确率（%）   |Top5准确率（%）      |Val准确率（%）     |Test准确率（%） |best threshold|
|:---:              |:---:          |:---:        |:---:            |:---:            |:---:              |:---:            |:---:          |:---:   
|[LFW]()            |[ResNet-50]()  |[ArcFace]()  |128              |98.047 (96.900)  |98.828 (98.860)    |-                |79.265 (78.916)|-            |
|[CASIA-WebFace]()  |[ResNet-50]()  |[ArcFace]()  |512              |99.219 (98.243)  |99.609 (99.445)    |-                |-              |-            |
|[VggFace2-clean]() |[ResNet-50]()  |[ArcFace]()  |512              |99.219 (99.365)  |100.000 (99.726)   |96.668 (95.849)  |-              |1.814 (1.786)|
|[VggFace2-part]()  |[ResNet-50]()  |[ArcFace]()  |512              |99.219 (98.772)  |100.000 (99.367)   |96.487 (94.119)  |-              |1.830 (1.764)|
|[VggFace2-sub]()   |[ResNet-50]()  |[ArcFace]()  |256              |99.219 (99.437)  |99.414 (99.590)    |95.463 (94.291)  |-              |1.650 (1.654)|
|[VggFace2-tiny]()  |[ResNet-50]()  |[ArcFace]()  |128              |99.609 (98.684)  |100.000 (99.859)   |83.718 (85.427)  |-              |1.661 (1.665)|
|[CSA-PEAL]()       |[ResNet-50]()  |[ArcFace]()  |128              |99.609 (96.746)  |100.000 (99.105)   |97.692 (98.438)  |-              |1.425 (1.462)|
|[CPLFW]()          |-              |-            |-                |-                |-                  |-                |-              |-            |
|[Vggface2_FP]()    |-              |-            |-                |-                |-                  |-                |-              |-            |



****

## Inference&推理部署

**以下是在不同设备不同模型上的推理情况**

|Model|Device|Inference Engine|TRT加速|(quant_aware_int8)量化|模型体积(MB)|时延(ms)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Backbone(ResNet50)|GPU TESLA v100|Paddle-Inference|Y|-|153|2.71|
|Backbone(ResNet50)|GPU GTX 1650Ti|Paddle-Inference|-|-|153|5.42|
|Backbone(ResNet50)|GPU MaxWell(jetson nano)|Paddle-Inference|-|-|153|48.42|
|Backbone(ResNet50)|CPU Raspberry Pi 4B|Paddle-Lite|-|-|153|243.22|
|Backbone(ResNet50)|CPU Raspberry Pi 4B|Paddle-Lite|-|Y|39|167.81|
|MTCNN|GPU TESLA v100|Paddle-Inference|-|-|-|14.13|
|MTCNN|GPU GTX 1650Ti|Paddle-Inference|-|-|-|15.56|
|MTCNN|GPU MaxWell(jetson nano)|Paddle-Inference|-|-|-|138.41|
|MTCNN|CPU Raspberry Pi 4B|Paddle-Lite|-|-|-|210.49|

### Paddle-Inference 部署

Paddle Inference为飞桨核心框架推理引擎。Paddle Inference功能特性丰富，性能优异，针对服务器端应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。

此项目通过Paddle Inference进行部署应用，在此Repo中提供了Python的使用样例

如果你对Paddle Inference有所疑惑，可以访问下面这个链接

[Paddle-Inference-API 文档](https://paddle-inference.readthedocs.io/en/latest/)


### Paddle-Lite 部署

Paddle Lite是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位支持包括移动端、嵌入式以及服务器端在内的多硬件平台。

当前Paddle Lite不仅在百度内部业务中得到全面应用，也成功支持了众多外部用户和企业的生产任务

此项目通过Paddle Lite进行部署应用，在此Repo中提供了Python的使用样例

如果你对此处有所疑惑，可以访问[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) , [Paddle-Lite's documentation](https://paddle-lite.readthedocs.io/zh/latest/index.html)

* Paddle-Lite-Inference-demo
* ├── FaceDatabase
* ├── main.py
* ├── model
* ├── MTCNN.py
* ├── README.md
* └── utils.py

### 量化

[PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/intro.html)
是一个专注于深度学习模型压缩的工具库，提供剪裁、量化、蒸馏、和模型结构搜索等模型压缩策略，帮助用户快速实现模型的小型化。

在此Repo中提供了PaddleSlim针对本模型进行量化的方法供大家参考
* quant
* ├── quant_post_dynamic.py #动态离线量化
* ├── quant_post_static.py #静态离线量化
* └── README.py
* 量化训练是最佳的量化方式，训练前可以在`config.py`中开启量化训练





