# 基于 PaddleDetection 的 SoccerNet 多目标跟踪基线
* 中文 | [English](./README_EN.md)

## 1. 介绍
* 一个基于 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 套件和 [SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) 数据集开发的足球和足球运动员多目标跟踪（MOT）的基线

* 包含 DeepSort、ByteTrack、JDE 和 FairMOT 四个经典的多目标跟踪模型，模型训练、评估、推理和部署全流程支持

## 2. 演示
* 足球与运动员多目标跟踪效果如下：

    ![demo](https://ai-studio-static-online.cdn.bcebos.com/c2e1a47da7c345c4b483367803b1c42c4bfba0984fa046c3ba19630687ac9398)

## 3. 目录
* 本项目目录结构如下：

    * configs -> 模型配置文件

    * dataset -> 数据集下载和预处理脚本

    * tools -> 评估脚本

## 4. 数据
### 4.1 数据简介
* SoccerNet Tracking 多目标跟踪数据集由来自主摄像机拍摄的 12 场完整足球比赛组成，包括：

    * 200 个视频剪辑，每段 30 秒，包含跟踪数据

    * 一个完整的半场视频，用跟踪数据标注

    * 12 场比赛的完整视频

### 4.2 数据格式
* 视频数据以视频帧图像的形式储存，样例如下：

    ![](https://ai-studio-static-online.cdn.bcebos.com/17cde91827304aeeaeceb946f025452796f52ac41d114657bfaadf591e977066)

* 真实值和检测结果存储在逗号分隔的 txt 文件中，共 10 列，样例如下：

    |帧 ID（Frame ID）|跟踪 ID（Track ID）|包围框左侧坐标（X）|包围框顶部坐标（Y）|包围框宽度（W）|包围框高度（H）|包围框置信度（Score）|未使用（-1）|未使用（-1）|未使用（-1）|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |1|1|914|855|55 |172|1|-1|-1|-1|
    |2|1|907|855|67 |172|1|-1|-1|-1|
    |3|1|901|855|79 |172|1|-1|-1|-1|
    |4|1|894|854|92 |173|1|-1|-1|-1|
    |5|1|888|854|104|173|1|-1|-1|-1|

* 数据集符合 MOT20 格式要求，由于某些值未使用，所以被标注为 -1

### 4.3 数据结构
* 数据集的目录结构如下：

    ```yaml
    - train
      - SNMOT-060
        - gt
          - gt.txt
        - img1
          - 000001.jpg
          ...
        - gameinfo.ini
        - seqinfo.ini
      ...
    - test
      ...
    - challenge
      ...
    ```

## 5. 模型
### 5.1 DeepSort
* 配置：[configs/snmot/deepsort](configs/snmot/deepsort)

* 论文：[Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

* 介绍：[DeepSORT](https://arxiv.org/abs/1812.00442) (Deep Cosine Metric Learning SORT) 扩展了原有的 [SORT](https://arxiv.org/abs/1703.07402) (Simple Online and Realtime Tracking) 算法，增加了一个 CNN 模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息，将检出的目标分配和更新到已有的对应轨迹上即进行一个 ReID 重识别任务。DeepSORT 所需的检测框可以由任意一个检测器来生成，然后读入保存的检测结果和视频图片即可进行跟踪预测。ReID 模型此处选择 [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 提供的`PCB+Pyramid ResNet101`和`PPLCNet`模型。

    ![](https://ai-studio-static-online.cdn.bcebos.com/4153812fdc6b44329e3d83d58991da6c846761ae04124c91950e0ccc17c85a99)
    
### 5.2 ByteTrack
* 配置：[configs/snmot/bytetrack](configs/snmot/bytetrack)

* 论文：[ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

* 介绍：[ByteTrack](https://arxiv.org/abs/2110.06864) 通过关联每个检测框来跟踪，而不仅是关联高分的检测框。对于低分数检测框会利用它们与轨迹片段的相似性来恢复真实对象并过滤掉背景检测框。此处提供了几个常用检测器的配置作为参考。由于训练数据集、输入尺度、训练 Epoch 数、NMS 阈值设置等的不同均会导致模型精度和性能的差异，请自行根据需求进行适配。
    
    ![](https://ai-studio-static-online.cdn.bcebos.com/e21a65dcd98242abb39854e33ebf46e21b55170e9c56461c8a058c3aed0b1eea)
    
    ![](https://ai-studio-static-online.cdn.bcebos.com/e7eb7077ef6e4934b0d7f336f634a983369ed91e89864b6299125fd0c230e76f)

### 5.3 JDE
* 配置：[configs/snmot/jde](configs/snmot/jde)

* 论文：[Towards Real-Time Multi-Object Tracking](https://arxiv.org/abs/1909.12605)

* 介绍：[JDE](https://arxiv.org/abs/1909.12605) (Joint Detection and Embedding) 是在一个单一的共享神经网络中同时学习目标检测任务和 Embedding 任务，并同时输出检测结果和对应的外观 Embedding 匹配的算法。JDE 原论文是基于 Anchor Base 的 YOLOv3 检测器新增加一个 ReID 分支学习 Embedding，训练过程被构建为一个多任务联合学习问题，兼顾精度和速度。

    ![](https://ai-studio-static-online.cdn.bcebos.com/4856babfbdeb44f88544a641bb67279d6b8fb59ac016492f9fd260c7a1b576c6)

### 5.4 FairMOT
* 配置：[configs/snmot/fairmot](configs/snmot/fairmot)

* 论文：[FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/abs/2004.01888v6)

* 介绍：[FairMOT](https://arxiv.org/abs/2004.01888v6) 以 Anchor Free 的 CenterNet 检测器为基础，克服了 Anchor Based 的检测框架中 Anchor 和特征不对齐问题，深浅层特征融合使得检测和 ReID 任务各自获得所需要的特征，并且使用低维度 ReID 特征，提出了一种由两个同质分支组成的简单 Baseline 来预测像素级目标得分和 ReID 特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

    ![](https://ai-studio-static-online.cdn.bcebos.com/790cbea0d7534b56b6e382f3f1daef41744a4155e55d449a96302c3952031f5c)

## 6. 使用
### 6.1 克隆安装
* 克隆本项目代码：

    ```bash
    $ git clone https://github.com/jm12138/SoccerNet_Tracking_PaddleDetection

    $ cd SoccerNet_Tracking_PaddleDetection

    $ pip install -r requirements.txt
    ```

* 克隆 PaddleDetection 代码：

    ```bash
    $ git clone https://github.com/PaddlePaddle/PaddleDetection
    
    $ cd PaddleDetection

    $ pip install -r requirements.txt
    ```

* 复制文件：

    ```bash
    $ cp -r ../configs ./
    $ cp -r ../dataset ./
    $ cp -r ../tools ./
    ```

### 6.2 数据下载
* 使用下载脚本可以快速下载数据：

    ```bash
    $ cd ./dataset/snmot
    
    $ python download_data.py
    ```

### 6.3 数据解压
* 使用如下命令解压数据：

    ```bash
    $ cd ../../
    $ mkdir ./dataset/snmot/SNMOT/images

    $ unzip -q ./dataset/snmot/SNMOT/tracking/train.zip -d ./dataset/snmot/SNMOT/images
    $ unzip -q ./dataset/snmot/SNMOT/tracking/test.zip -d ./dataset/snmot/SNMOT/images
    $ unzip -q ./dataset/snmot/SNMOT/tracking/challenge.zip -d ./dataset/snmot/SNMOT/images
    ```

### 6.4 数据处理
* 转换数据格式以符合 PaddleDetection 的要求：

    ```bash
    $ cd ./dataset/snmot
    
    $ python gen_labels.py
    $ python gen_image_list.py
    $ python gen_det_coco.py
    $ python gen_det_results.py
    $ python zip_gt.py
    ```

### 6.5 模型训练
* 指定一个模型配置文件，使用如下命令进行模型训练（以 FairMOT 为例）：

    ```bash
    $ cd ../../

    $ python tools/train.py -c ./configs/snmot/fairmot/fairmot_dla34_30e_1088x608.yml 
    ```

### 6.6 模型验证
* 使用如下命令进行模型评估：

    ```bash
    $ python tools/eval_mot.py \
        -c ./configs/snmot/fairmot/fairmot_dla34_30e_1088x608.yml \
        -o weights=./output/fairmot_dla34_30e_1088x608/model_final

    $ cd ./output/mot_results
    
    $ zip soccernet_mot_results.zip *.txt

    $ cd ../../
    
    $ python tools/evaluate_soccernet_v3_tracking.py \
        --BENCHMARK SNMOT \
        --DO_PREPROC False \
        --SEQMAP_FILE tools/SNMOT-test.txt \
        --TRACKERS_TO_EVAL test \
        --SPLIT_TO_EVAL test \
        --OUTPUT_SUB_FOLDER eval_results \
        --TRACKERS_FOLDER_ZIP ./output/mot_results/soccernet_mot_results.zip \
        --GT_FOLDER_ZIP ./dataset/snmot/gt.zip
    ```

### 6.7 模型推理
* 使用如下命令进行模型推理：

    ```bash
    $ python tools/infer_mot.py \
        -c ./configs/snmot/fairmot/fairmot_dla34_30e_1088x608.yml \
        -o weights=./output/fairmot_dla34_30e_1088x608/model_final \
        --image_dir ./dataset/snmot/SNMOT/images/challenge/SNMOT-021/img1 \
        --frame_rate 25 \
        --output_dir ./output \
        --save_videos
    ```

### 6.8 更多详情
* 模型部署和更多使用指南请参考 PaddleDetection 官方文档

## 7. 参考
* 相关链接：

    * [SoccerNet: Soccer Video Understanding Benchmark Suite](https://www.soccer-net.org/home)

    * [SoccerNet Tracking Official Website](https://www.soccer-net.org/tasks/tracking)

    * [SoccerNet Tracking Development Kit](https://github.com/SoccerNet/sn-tracking)

    * [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

    * [多目标跟踪（MOT）入门](https://zhuanlan.zhihu.com/p/97449724)

* 论文引用：

    ```BibTeX
    @inproceedings{Wojke2017simple,
        title={Simple Online and Realtime Tracking with a Deep Association Metric},
        author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
        booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
        year={2017},
        pages={3645--3649},
        organization={IEEE},
        doi={10.1109/ICIP.2017.8296962}
    }

    @article{DBLP:journals/corr/abs-2110-06864,
        title = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
        author = {Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
        doi = {10.48550/ARXIV.2110.06864},
        url = {https://arxiv.org/abs/2110.06864},
        publisher = {arXiv},
        year = {2021}
    }

    @article{wang2019towards,
        title={Towards Real-Time Multi-Object Tracking},
        author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
        journal={arXiv preprint arXiv:1909.12605},
        year={2019}
    }

    @article{zhang2020fair,
        title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
        author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
        journal={arXiv preprint arXiv:2004.01888},
        year={2020}
    }
    ```
