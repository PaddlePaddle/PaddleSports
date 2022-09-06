# 导包
import paddle.nn as nn
from aagcn_backbone import AAGCN_backbone
from aagcn_head import STGCNHead


class AAGCN_framework(nn.Layer):
    def __init__(self,num_classes = 30):
        super().__init__()
        self.backbone = AAGCN_backbone()
        self.head = STGCNHead(num_classes = num_classes)
    def forward(self,data):
        feature = self.backbone(data)
        cls_score = self.head(feature)
        return cls_score
