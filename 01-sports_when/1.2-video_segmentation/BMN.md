# BMN 视频动作定位模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)



## 模型简介

BMN模型是百度自研，2019年ActivityNet夺冠方案，为视频动作定位问题中proposal的生成提供高效的解决方案，在PaddlePaddle上首次开源。此模型引入边界匹配(Boundary-Matching, BM)机制来评估proposal的置信度，按照proposal开始边界的位置及其长度将所有可能存在的proposal组合成一个二维的BM置信度图，图中每个点的数值代表其所对应的proposal的置信度分数。网络由三个模块组成，基础模块作为主干网络处理输入的特征序列，TEM模块预测每一个时序位置属于动作开始、动作结束的概率，PEM模块生成BM置信度图。

AI Studio项目使用链接：[ActivityNet Challenge 2019 冠军模型：BMN](https://aistudio.baidu.com/aistudio/projectdetail/2250674?contributionType=1)

<p align="center">
<img src="https://raw.githubusercontent.com/FeixiangLu/PaddleSports/main/01-sports_when/1.2-video_segmentation/images/BMN.png" height=300 width=400 hspace='10'/> <br />
BMN Overview
</p>

## 环境配置
详见[配置清单](./env.md)

## 数据准备

### 通用视频数据集 
ActivityNet [数据集准备](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/ActivityNet.md)

### 体育视频数据集
不同于ActivityNet，FineAction等数据集，我们采用了更精细的动作数据集--乒乓球转播画面，该数据集具有动作时间跨度短，分布密集等特点，给传统模型精确定位细粒度动作带来了很大挑战。
<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/d69f38b235934411ac9b8f7d24a8d11a070c57780dd84eaaa742ed6c170956a1" width="600" height="400" /> <br>
视频来源：2020东京奥运会乒乓球男单决赛
</p>  

#### 数据下载
数据集可以从AIsutudio乒乓球时序动作定位大赛官方页面[[训练集](https://aistudio.baidu.com/aistudio/datasetdetail/122998)][[测试集A](https://aistudio.baidu.com/aistudio/datasetdetail/123004)][[测试集B](https://aistudio.baidu.com/aistudio/datasetdetail/123009)]。

该数据集包含了19-21赛季兵乓球国际比赛（世界杯、世锦赛、亚锦赛，奥运会）和国内比赛（全运会，乒超联赛）中标准单机位高清转播画面的特征信息，共包含912条视频特征文件，每个视频时长在0～6分钟不等，特征维度为2048，以pkl格式保存。我们对特征数据中面朝镜头的运动员的回合内挥拍动作进行了标注，单个动作时常在0～2秒不等，训练数据为729条标注视频，A测数据为91条视频，B测数据为92条视频，训练数据标签以json格式给出。

#### 训练数据 
包含912条PP-TSM抽取的视频特征，特征保存为pkl格式，文件名对应视频名称，读取pkl之后以(num_of_frames, 2048)向量形式代表单个视频特征。
```python
{'image_feature': array([[-0.00178786, -0.00247065,  0.00754537, ..., -0.00248864,
        -0.00233971,  0.00536158],
       [-0.00212389, -0.00323782,  0.0198264 , ...,  0.00029546,
        -0.00265382,  0.01696528],
       [-0.00230571, -0.00363361,  0.01017699, ...,  0.00989012,
        -0.00283369,  0.01878656],
       ...,
       [-0.00126995,  0.01113492, -0.00036558, ...,  0.00343453,
        -0.00191288, -0.00117079],
       [-0.00129959,  0.01329842,  0.00051888, ...,  0.01843636,
        -0.00191984, -0.00067066],
       [-0.00134973,  0.02784026, -0.00212213, ...,  0.05027904,
        -0.00198008, -0.00054018]], dtype=float32)}
<class 'numpy.ndarray'>
```

#### 训练及测试标签
```javascript
# label_cls14_train.json
{
    'fps': 25,    #视频帧率
    'gts': [
        {
            'url': 'name_of_clip.mp4',      #名称
            'total_frames': 6341,    #总帧数
            'actions': [
                {
                    "label_ids": [7],   #动作类型编号
                    "label_names": ["name_of_action"],     #动作类型
                    "start_id": 201,  #动作起始时间,单位为秒
                    "end_id": 111    #动作结束时间,单位为秒
                },
                ...
            ]
        },
        ...
    ]
}

# label_cls14_A.json & label_cls14_B.json
{
    'fps': 25,    #视频帧率
    'gts': [
        {
            'url': 'name_of_clip.mp4',      #名称
            'total_frames': 6341,    #总帧数
            'actions': []   #空置
        },
        ...
    ]
}
```

#### 数据预处理
运行脚本get_instance_for_bmn.py，提取二分类的proposal，windows=8，根据gts和特征得到BMN训练所需要的数据集:
```javascript
#数据格式
{
  "5679b8ad4eac486cbac82c4c496e278d_133.56_141.56": {     #视频名称_片段起始时间_片段结束时间(s)
          "duration_second": 8.0,
          "duration_frame": 200,
          "feature_frame": 200,
          "subset": "train",
          "annotations": [
              {
                  "segment": [
                      6.36,#动作起始时间
                      8.0  #动作结束时间
                  ],
                  "label": "11.0",
                  "label_name": "普通"
              }
          ]
      },
      ...
}
```
在/PaddleVideo/applications/TableTennis/目录下，
- 运行val_split.py生成验证集；
- 运行get_instance_for_bmn.py生成bmn训练数据和标签；
- 运行fix_bad_label.py矫正标签和数据是否一一对应，数据中一些无对应标签的feature将不参与训练。


## 模型训练

数据准备完毕后，可以通过如下方式启动训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_bmn main.py  --validate -c PaddleVideo/applications/TableTennis/configs/bmn_tabletennis.yaml
```

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型

### 单卡训练

单卡训练请将配置文件中的`DATASET.batch_size`字段修改为合适大小（可根据实际资源和超参数设置），如下:

```yaml
DATASET: #DATASET field
  batch_size: 4 #single card bacth size
```

单卡训练启动方式如下:

```bash
python -B main.py  --validate -c PaddleVideo/applications/TableTennis/configs/bmn_tabletennis.yaml
```


## 模型测试

可通过如下方式进行模型测试:

```bash
python main.py --test -c configs/localization/bmn.yaml -w output/BMN/BMN_epoch_00009.pdparams -o DATASET.test_batch_size=1
```

- 目前仅支持**单卡**， `batch_size`为**1**进行模型测试，

- 请使用测试标签文件label_gts_A.json和label_gts_B.json，并通过`METRIC.ground_truth_filename`字段指定该文件，

- 通过 `-w`参数指定待测试模型文件的路径，您可以下载我们训练好的模型进行测试[BMN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams)

- 上述程序会将运行结果保存在配置文件`METRIC.output_path`字段指定的路径，默认为`data/bmn/BMN_Test_output`文件夹下，测试结果保存在配置文件`METRIC.result_path`字段指定的文件，默认为`data/bmn/BMN_Test_results/bmn_results_validation.json`文件。

- 我们基于ActivityNet官方提供的测试脚本，计算AR@AN和AUC。具体计算过程请参考[anet_prop.py](https://github.com/PaddlePaddle/PaddleVideo/blob/main/paddlevideo/metrics/ActivityNet/anet_prop.py)文件。

- 注：评估时可能会出现loss为nan的情况。这是由于评估时用的是单个样本，可能存在没有iou>0.6的样本，所以为nan，对最终的评估结果没有影响。

在A测数据集下评估AUC精度为19，大家可根据使用的数据集调整训练策略，提升精度。


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/localization/bmn.yaml \
                                -p data/BMN.pdparams \
                                -o inference/BMN
```

上述命令将生成预测所需的模型结构文件`BMN.pdmodel`和模型权重文件`BMN.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example_feat.list \
                           --config configs/localization/bmn.yaml \
                           --model_file inference/BMN/BMN.pdmodel \
                           --params_file inference/BMN/BMN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

- `input_file`为文本文件，指定待推断的文件信息，包括特征文件路径`feat_path`和视频时长(单位:s)`duration_second`。

输出示例如下:

```
BMN Inference results of data/example_feat.npy :
{'score': 0.7968077063560486, 'segment': [0.0, 122.9877]}
{'score': 0.49097609519958496, 'segment': [12.423000000000002, 124.23]}
{'score': 0.21395835280418396, 'segment': [39.7536, 122.9877]}
{'score': 0.2106524258852005, 'segment': [0.0, 109.3224]}
{'score': 0.06876271963119507, 'segment': [23.6037, 114.2916]}
```

- 默认只打印前5个得分最高的proposal，所有的预测结果可在输出文件中查看，默认输出文件路径为`data/bmn/BMN_INFERENCE_results`。输出路径可在配置文件中的`INFERENCE.result_path`自行修改。

## 参考论文

- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
