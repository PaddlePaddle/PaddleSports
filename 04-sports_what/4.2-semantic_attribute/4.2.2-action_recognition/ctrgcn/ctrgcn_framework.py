# 导包
import paddle.nn as nn
from ctrgcn_backbone import CTRGCN_backbone
from ctrgcn_head import CTRGCNHead


class CTRGCN_framework(nn.Layer):
    def __init__(self,num_classes = 30):
        super().__init__()
        self.backbone = CTRGCN_backbone()
        self.head = CTRGCNHead(num_classes = num_classes)
    def forward(self,data):
        feature = self.backbone(data)
        cls_score = self.head(feature)
        return cls_score
