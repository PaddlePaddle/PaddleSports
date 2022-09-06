# 静态的运动场景中人物三维渲染

## 1. 项目背景

- 在体育比赛时，如果能360°的展示一个选手的精彩运动瞬间，那将是一个非常炫酷和赏心悦目的事
- 本项目使用《舞动风暴》节目中的风暴时刻的数据，以及**COLMAP、PP-Humanseg、NeRF**等方法来展示如何对静态的运动场景中人物进行**任意视角的三维渲染**

## 2. 效果展示

- 本项目的效果如下。由于运动场景中主角是人，所以使用PP-Humanseg将人像自动抠出，但由于**部分图像背景剔除不干净**，所以导致有的视角渲染的效果不佳
- 本项目有在线体验版，AI Studio链接：https://aistudio.baidu.com/aistudio/projectdetail/4439534

| 视角一| 视角二|
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/aaeeeeab1f2d4cefb42186dafec5cb59b27d35934bc744ed95652241e471301f)| ![](https://ai-studio-static-online.cdn.bcebos.com/5ae034f5ab5c4c21afe354cf7043544e9a2bfcc51a2943b9952382f35fac356b)|

## 3. 数据准备

### 3.1 数据获取

- 本项目所用数据为《舞动风暴》节目中的[风暴时刻的片段](https://www.bilibili.com/video/BV1mJ411F7Dj?spm_id_from=333.337.search-card.all.click&vd_source=f8c967257e5e66e7ab9fdb724afa02f7)，片段是180°的相机布设拍摄得到的视频，从视频中获取图像数据
- 从B站中选用的视频片段为黄琛迪的爱不释手，部分图片如下所示

| 正面 | 侧面 |
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/e221425bf834435d8210e6dfc59381a3c4eed98853814e719c6f0e9f46099d5e) |![](https://ai-studio-static-online.cdn.bcebos.com/2806219f21b94e28ac92d0a91b5ea31a7428285a94d74f66acad9d4d94c1312f) |

### 3.2 像片位姿获取

- 使用[COLMAP](https://demuc.de/colmap/)对像片的姿态进行获取，具体使用步骤克参考AI Studio项目[COLMAP+NeRF，使用自己手机拍摄的2D照片3D渲染](https://aistudio.baidu.com/aistudio/projectdetail/4132862?contributionType=1)
- 使用COLMAP进行稀疏重建，解算出每张像片的位姿，示意图如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/506ce7ad73eb4845ad6da53ad7d40608323fa4cdfdde479aa2f1f23ad9a10f5c)

**注**：获取的图像以及位姿数据已经压缩为`Dance.zip`文件，并上传到work文件夹下

### 3.3 PP-Humanseg抠出人像

- 使用PaddleSeg提供的[PP-Humanseg](https://gitee.com/paddlepaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg)模型处理图像数据，剔除背景并替换为黑色，具体步骤如下：

1. 克隆仓库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

2. 安装依赖

```shell
cd PaddleSeg
pip install -r requirements.txt
```

3. 下载所需要的权重

```shell
cd ./contrib/PP-HumanSeg

wegt https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_inference_model_with_softmax.zip
```

4. 执行命令抠图

- 将本文件夹下的`my_seg_demo.py`文件放于`PaddleSeg/contrib/PP-HumanSeg/src/`文件夹下，然后执行以下脚本
```shell
python ./PaddleSeg/contrib/PP-HumanSeg/src/my_seg_demo.py \
  --config PaddleSeg/contrib/PP-HumanSeg/inference_model/human_pp_humansegv1_server_512x512_inference_model_with_softmax/deploy.yaml \
  --folder_path ./work/Dance/images/ \
  --save_dir ./work/Dance/images_result
```

- 其中`folder_path`为待抠图的文件夹路径，`save_dir`为保存结果的文件夹路径

5. 抠图效果展示

| 正面 | 侧面 |
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/168ad8b4a4444acfbda103c03c3e395d83e2652536e448019b852fb87debf249) | ![](https://ai-studio-static-online.cdn.bcebos.com/6a4201641eb8425991d9588ebb787d5c59c223625fea4eeb9896bfaed4bebe72) |

- **注**：可以看到效果还可以。之所以没有先抠图再获取位姿，是因为进行稀疏重建时是需要特征点匹配的，如果先抠图会导致提取出来的特征点比较少

### 3.4 生成LIFF格式数据

- 获取位姿之后，要进行NeRF训练，还需要将其转换为可训练的数据格式，本项目选择的是LIFF格式

```shell
# 进入当前文件夹，配置环境，除了paddlepaddle-gpu==2.2.2，还需安装requirements.txt的
pip install -r requirements.txt
```

- 执行以下脚本，生成LIFF数据

```shell
python gen_poses.py --scenedir ./data/nerf_llff_data/Dance
```

**注**：处理好后的数据已经放在当前文件夹下的`/data/nerf_llff_data/Dance`文件夹中

## 4. 2D转3D的NeRF渲染

- 在生成训练所需要的LIFF数据之后，使用[NeRF](https://www.matthewtancik.com/nerf)对该场景的隐式神经辐射场训练
- 训练所用的配置文件在`configs`文件夹下，为`dance.txt`文件

### 4.1 训练NeRF
- 执行以下脚本，训练Dance场景

```shell
python run_nerf.py --config configs/dance.txt
```

### 4.2 测试NeRF

- 执行以下脚本，测试训练了200k次iteration的权重效果
- 由于该场景增加了渲染时细采样的次数，时间较久，约40分钟

```shell
python run_nerf.py --config configs/dance.txt --ft_path ./logs/dance_test/200000.pdparams --render_only
```

### 4.3 渲染特定位姿的图像

- NeRF训练好的隐式神经辐射场，是可以从任意视角渲染该场景，因此本项目增加一个脚本，可以**指定要渲染的照片其拍摄的位姿，就能生成该位姿下的图像**
- 单张影像的位姿数据为3×5的矩阵，包括3×3的旋转矩阵、3×1的位移矩阵以及3×1的相机内参，其中**相机内参是确定的**，其他参数可以自己设置
- 将位姿数据保存为.txt格式即可使用`test_render_poses.py`脚本生成该场景下特定位姿下拍照的图像，使用示例如下：

```shell
python test_render_poses.py --config configs/dance.txt --ft_path ./logs/dance_test/200000.pdparams --render_only \
--rdpose_path ../poses/poses_21.txt
```
- 其中，`rdpose_path`参数为位姿文件的所在路径

## 5. 项目总结

- 本项目通过COLMAP获取照片位姿，PP-Humanseg抠出人物，NeRF进行2D转3D的渲染，基本能达到静态的运动场景中人物三维渲染的目的
- **不足之处**：
    - 由于图像中背景部分有颜色和躯体很接近，所以抠人像出来的时候，并没有能很好的抠出干净的人像，导致**训练的NeRF场景有部分视角不是很清晰**
    - 训练的数据是由180°布设的相机拍摄的照片，而想推广到现实中的应用，需要**做到单目、或者稀疏相机拍摄的图像就能够生成相应的渲染图**，最初版本的NeRF并不满足此需求。但是近段时间开源一些优秀作品，如[HumanNeRF](https://grail.cs.washington.edu/projects/humannerf/)有很大可能完成这项任务，下一阶段是把这些优秀的作品使用paddle复现出来
