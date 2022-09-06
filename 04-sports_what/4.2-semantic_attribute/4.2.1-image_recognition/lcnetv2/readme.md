# LCNetV2进行场景识别

骨干网络对计算机视觉下游任务的影响不言而喻，不仅对下游模型的性能影响很大，而且模型效率也极大地受此影响，但现有的大多骨干网络在真实应用中的效率并不理想，特别是缺乏针对 Intel CPU 平台所优化的骨干网络，我们测试了现有的主流轻量级模型，发现在 Intel CPU 平台上的效率并不理想，然而目前 Intel CPU 平台在工业界仍有大量使用场景，因此我们提出了 PP-LCNet 系列模型，PP-LCNetV2 是在 [PP-LCNetV1](./PP-LCNet.md) 基础上所改进的。

> [LCNet v2场景识别封装版](https://aistudio.baidu.com/aistudio/projectdetail/4447604)该ai studio 项目为LCNet v2封装版

>[PPSIG:paddlesports LCNet v2场景识别](https://aistudio.baidu.com/aistudio/projectdetail/4446789)该ai studio 项目为LCNet v2解释版

>参考github paddleclas:
>https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md
## 1. 数据集展示：
![](https://ai-studio-static-online.cdn.bcebos.com/d32b798fc84a41d489b12d0ebe8350d2095704f1a809485da3a35a49d7067227)
>数据集使用sports10(主要为体育游戏图片)，来自论文Contrastive Learning of Generalized Game Representations，论文github地址：https://github.com/ChintanTrivedi/contrastive-game-representations 代码为tensorflow，数据集提供。

## 2. 模型训练及验证

### main.py配置参数介绍

```python
    parser.add_argument('--class_num',type=int,default=10)
    parser.add_argument('--epoches',type=int,default=100)
    parser.add_argument('--use_pretrained', type=str,default= "")#PPLCNetV2_base_pretrained_without_fc.pdparams
    parser.add_argument('--data_dir', type=str,default="./data/d/Sports10/")
    parser.add_argument('--BATCH_SIZE', type=int,default=32) 
    parser.add_argument('--load_pretrain_model', type=str,default="")
    parser.add_argument('--output_model_dir', type=str,default="model")
    parser.add_argument('--is_train',type=int,default=1)
    parser.add_argument("--output_log_dir",type=str,default="./log")

```

1. class_num为分类数
2. epoches为训练轮数
3. use_pretrained 为LCNet backbone是否使用预训练权重
4. data_dir为数据集的文件夹
5. load_pretrain_model为需要导入的模型参数文件路径
6. output_model_dir为训练模型参数文件输出路径
7. is_train如果为1，则代表模型训练，如果不为1，则代表模型验证
8. output_log_dir,为log输出路径

>训练的学习率优化器可以在lcnet_main.py中自行修改，

>运行示例：!python main.py --epoches 20 --use_pretrained PPLCNetV2_base_pretrained_without_fc.pdparams --BATCH_SIZE 128 --is_train 1 --load_pretrain_model 0.94.pdparams

>参数文件大小23.8MB,可在模型验证集准确率为0.94.



如果需要使用对比学习可参考[PPSIG:对比学习在监督学习中应用（分类任务） - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/4358899)项目



## 3. 单张图片检验

测试图片
![](https://ai-studio-static-online.cdn.bcebos.com/04fed3410a4f4d12b3fd7a285ef1b373853c440201434c74aadbcaec17d72ae8)





```python 
from lcnet_framework import PPLCNetV2_model
import paddle
from paddle.vision.transforms import Resize
import cv2
import numpy as np

model = PPLCNetV2_model(10)
model.set_state_dict(paddle.load("0.94.pdparams"))

model.eval()
img_A = cv2.cvtColor(cv2.imread("nba2k12117.jpg", flags=cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)#内容图
g_input = Resize(size = (360,640))(img_A)
g_input = g_input[np.newaxis, ...].transpose(0, 3, 1, 2)  # NHWC -> NCHW
g_input = paddle.to_tensor(g_input)
g_input = g_input.astype("float32")/127.5-1
# print(g_input)
classid = paddle.argmax(model(g_input),axis = 1)
# print(classid)
class2id = {
            "AmericanFootball":0,
            "Basketball":1,
            "BikeRacing":2,
            "CarRacing":3,
            "Fighting":4,
            "Hockey":5,
            "Soccer":6,
            "TableTennis":7,
            "Tennis":8,
            "Volleyball":9
        } 
id2class = sorted(class2id.items(),key = lambda x:x[1])
id2class= [i[0] for i in id2class]
print("该图片属于",id2class[classid],"类") #该图片属于 Basketball 类


```

