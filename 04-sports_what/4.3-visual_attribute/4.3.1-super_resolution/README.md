# 使用Real-ESRGAN恢复低分辨率的足球场景照片

## 1. Real-ESRGAN简介

- [Real-ESRGAN](https://arxiv.org/abs/2107.10833)采用纯合成数据对ESRGAN朝着真实复原应用进行扩展，得到了本项目所提的Real-ESRGAN，这是ESRAGN、EDVR等超分领域里程碑式论文作者的又一力作
- Real-ESRGAN构建了一个**高阶退化建模过程**以更好的模拟复杂的真实退化。在合成过程中，同时还考虑的重建的ringing, overshoot伪影等问题

### 1.1 高阶退化建模

- 经典的退化模型仅仅包含固定数量的基本退化，这种退化可以视作**一阶退化**
- 然而，**实际生活中的退化过程非常多样性**，通常包含不同的处理，比如成像系统、图像编辑、网络传输等等。当我们想要对网络下载的低分辨率图像进行重建时，难度就很大了，例如会遇到以下问题：
    - 原始图像可能由多年前的手机拍摄所得，包含严重的退化问题
    - 当图像通过锐化软件编辑处理后又会引入overshoot以及模糊伪影等问题
    - 当图像经过网络传输后又会进一步引入不可预料的压缩噪声等

- 上述的这些操作都会使得退化变得很复杂，为缓解上述问题，Real-ESRGAN提出了**高阶退化模型**：它包含多个重复退化的过程，其中，每个阶段采用相同的退化处理但具有不同的退化超参，流程如下图所示（**注：高阶退化模型仍然不完美，无法覆盖真实世界的整个退化空间**。相反，它通过修改数据合成过程对已有盲图像超分的退化边界进行扩展）：

![](https://ai-studio-static-online.cdn.bcebos.com/756e5991803b45079ab0ef7362a4fb844af89dce7ca346c4993a5aa8f65acaec)

### 1.2 Real-ESRGAN效果

- 将Real-ESRGAN的torch权重转为paddle的权重，对手上有的**真实的低分辨率数据**（行车记录仪影像、300m高度飞行的DJI无人机影像）进行超分重建，结果如下图所示：

| 低分辨率影像 | Real-ESRGAN超分 |
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/060ea2ff925e444f877643f8b32b67590cd08115247b460eae785d0a0547c2d7)|![](https://ai-studio-static-online.cdn.bcebos.com/c51719e8a80148cdbc7edf4452a4e391749b3533503a4eb0943afcac99defeac) |
| ![](https://ai-studio-static-online.cdn.bcebos.com/7e7896a0865e4f54ad1d4cba637cc8fb557713f57c034d1d8a6dfb32f344bf86) | ![](https://ai-studio-static-online.cdn.bcebos.com/82af34f06cfe42b3a86bb334d3fb758687385c47403743bd837ef9320e7c63f3)|

### 1.3 对足球场景超分的不足

- Real-ESRGAN对包浆的自然影像的处理效果很好，但是对足球场景的图像通常会有一些**失真**的效果，例如**虚化的背景球员恢复出来扭曲的效果、球员的肢体皮肤出现不真实的伪影等**，选取了一些百度图片上的低分辨率足球场景图像块测试，如下所示：

| 低分辨图像 | Real-ESRGAN超分|
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/4454dbc0f9234f4a9d579d5f3ead4a8578d4e1b953ea44fb975e38600dfbca32)| ![](https://ai-studio-static-online.cdn.bcebos.com/e8f25869ad364d78b6ba7c87901d5e52ee0dd7c5fdee442e9d1a5bd6c512564f) |
| ![](https://ai-studio-static-online.cdn.bcebos.com/3c72f38ace444b7f9365bfc630e2e99effc005e4e39d4328b98e95cf57ec85b9)| ![](https://ai-studio-static-online.cdn.bcebos.com/5c82f0b08a12460e8f4c6bbb9d4842e96bf50f32ff0b4f69a6d53f30fa86dff2)|

- 由于Real-ESRGAN训练时没有针对足球场景的数据进行训练，所以本项目**使用paddle版本的Real-ESRGAN以及2022年欧冠决赛的高清视频的关键帧训练**，以期缓解此种现象

## 2. 足球场景图像训练结果

- 将finetune训练后的模型与预训练对比，可以看到有一些改善，**没那么多不真实的伪影**

| 低分辨率图像 | origin Real-ESRGAN| finetune Real-ESRGAN |
| --- | --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/8ee152e550434f2bbb029688dbe919cb6c3d15c206da42389ae71caf2e18f03b)| ![](https://ai-studio-static-online.cdn.bcebos.com/fee89d442b574ba38fa76ce847c5e388562030b2791848079b5b521b0641fd64)| ![](https://ai-studio-static-online.cdn.bcebos.com/4ecd6721ecad41e59d0a920a5c12fb77492642a200c04a4e88d69246b82ddc3f)|
|![](https://ai-studio-static-online.cdn.bcebos.com/e8dbb4011b9343568c23f02ad58e4d54228c47caf86e48029c0332199613a333) | ![](https://ai-studio-static-online.cdn.bcebos.com/3fd977ed38ff4e14955eed31c313ef2d660fa579061548c0a4271eee9206db20)| ![](https://ai-studio-static-online.cdn.bcebos.com/c8453dc0fd4f45a79779d5d123d65bea8aaf8404b85246c3aa1ba2ec1f3567e7)|
| ![](https://ai-studio-static-online.cdn.bcebos.com/b85000955c10449d9265131f49145f10490fb25b4c35421cb575e19c765d1aef)|![](https://ai-studio-static-online.cdn.bcebos.com/baf6c033bb5247d789c702a209d0dcd00410f6d7aab34deeb10fa35f9ad8624e) |![](https://ai-studio-static-online.cdn.bcebos.com/e9609d04fd5a4fddaa580d3f69b5c85ed07a8670595748d89cd1f9a2d44e6ca8) |
|![](https://ai-studio-static-online.cdn.bcebos.com/193b4c5f4a254926a7d591d3d44e73c5b835891552654965996a1a60be3bd499) | ![](https://ai-studio-static-online.cdn.bcebos.com/22ee703c6f9f4814bf5b7558cb6ae7d5ec9a405e543f4fb592fba5be5042f447)| ![](https://ai-studio-static-online.cdn.bcebos.com/1854e09c6082432681d44987222f6619d19a14f37e494f2787ea6cf7d19a7c42)|
|![](https://ai-studio-static-online.cdn.bcebos.com/a834cc9e86b9478da5b8701676a81d95e9d0e5bf0aea4236ab4bcc6f7d427e98) | ![](https://ai-studio-static-online.cdn.bcebos.com/d70a45fad3744b339f1e3373b1fc0bec1e647d80e7bb4ee2addd225a17f85614)| ![](https://ai-studio-static-online.cdn.bcebos.com/75f2dcecb4324c28b83884c7cff71204223386096c854444b84bc002bc1674f2)|

## 3. 数据集、预训练模型、文件结构

### 3.1 数据集

- Real-ESRGAN模型训练并不需要事先准备低分辨率-高分辨率图像对，而是**只需提供高分辨率图像**，在训练的过程中，会使用**高阶退化方式**将高分辨率图像退化为低分辨率的图像构成图像对训练
- 本项目训练所用的高分辨率数据为`2022年欧冠决赛的高清视频的关键帧`，使用`ffmpeg`提取的关键帧为高分辨率图像，如何操作见Ai Studio项目地址：https://aistudio.baidu.com/aistudio/projectdetail/4320652


- 将所有数据处理到对应的文件夹中，使用以下脚本生成meta_info文件：

```shell
python tools/generate_meta_info.py --input datasets/SoccerHR_up datasets/SoccerHR_sq datasets/SoccerHR_sh/ datasets/SoccerHR_down/ \
--root datasets/ datasets/ datasets/ datasets/ --meta_info datasets/meta_Soccer.txt
```

### 3.2 预训练模型

- real-ESRGAN的torch权重转为paddle的权重，并以此作为预训练模型finetune，名为`RealESRGAN_x4plus.pdparams`。百度网盘：[下载链接](https://pan.baidu.com/s/1hoWrlaEDnDiT-iLFiuvhaw)，提取码：t6sl

- finetune训练102500次（约82小时）之后的权重，名为`net_g_102500.pdparams`。百度网盘：[下载链接](https://pan.baidu.com/s/1TsVyXdJZAwaXYZMLEVRBfQ)，提取码：hoqd

### 3.3 文件结构

```
4.3.1-super_resolution
    |-- data                       # 数据集相关文件
    |-- loss                       # 损失函数
    |-- models                     # 模型相关文件
    |-- options                    # 训练配置文件
    |-- tools                      # 一些工具代码
    |-- utils                      # 一些工具代码
    |-- README.md                  # README.md文件
    |-- train.py                   # 训练代码
    |-- make_lq_data.py            # 制作低分辨率影像数据代码
    |-- requirements.txt           # 依赖库安装文件
```

## 4. 环境依赖

- PaddlePaddle >= 2.3.0
- ppgan
- opencv-python

## 5. 快速开始

### 5.1 单卡训练(V100-32G)

1. 使用real-ESRGAN模型的权重作为初始化权重，
`RealESRGAN_x4plus.pdparams`进行训练.

2. 修改配置文件`options/train_realesrgan_x4plus.yml` ：

- 数据集部分
   ```
   datasets:
    train:
        name: DF2K_Anime_OST
        type: RealESRGANDataset
        dataroot_gt: datasets
        meta_info: datasets/meta_Soccer.txt
        io_backend:
        type: disk
        io_backend:
            type: disk
   ```

- 预训练权重路径部分
   ```
    path:
        # use the pre-trained Real-ESRGAN model
        experiments_root: experiments
        pretrain_network_g: ../weights/RealESRGAN_x4plus.pdparams
        param_key_g: params_ema
        strict_load_g: true
        resume_state: ~
   ```

3. 训练 Real-ESRGAN：

   ```
   python train.py --yml_path options/train_realesrgan_x4plus.yml
   ```

### 5.2 模型测试

使用训练了102500个iteration的模型进行测试

```shell
python tools/predict.py --input inputs --output results --model_path experiments/train_RealESRGANx4plus_400k_B23G4/models/net_g_102500.pdparams --block 23
```

- 输出图片在 `results`文件夹下，也可以修改换成其他保存路径

### 5.3 模型导出

模型动转静导出：

```shell
python tools/export_model.py --model-dir ./experiments/train_RealESRGANx4plus_400k_B23G4/models/net_g_102500.pdparams --save-inference-dir ./infer/ --block 23
```

运行顺利则最终在`infer/`文件夹下会生成下面的3个文件：

```
infer
  |----inference.pdiparams     : 模型参数文件
  |----inference.pdmodel       : 模型结构文件
  |----inference.pdiparams.info: 模型参数信息文件
```

### 5.4 模型推理

```shell
python tools/infer.py --input inputs --output results_infer --model_path ./infer
```

运行顺利则输出图片在 `results_infer`文件夹下
