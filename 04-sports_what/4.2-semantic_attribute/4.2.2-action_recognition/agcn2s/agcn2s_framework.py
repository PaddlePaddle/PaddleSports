# 导包
import paddle.nn as nn
from agcn2s_backbone import AGCN2S_backbone
from agcn2s_head import STGCNHead


class AGCN2S_framework(nn.Layer):
    def __init__(self,num_classes = 30):
        super().__init__()
        self.backbone = AGCN2S_backbone()
        self.head = STGCNHead(num_classes = num_classes)
    def forward(self,data):
        feature = self.backbone(data)
        cls_score = self.head(feature)
        return cls_score
