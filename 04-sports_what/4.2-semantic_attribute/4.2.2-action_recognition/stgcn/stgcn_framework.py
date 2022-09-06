# 导包
import paddle.nn as nn
from stgcn_backbone import STGCN_BackBone
from stgcn_head import STGCNHead

class STGCN_framework(nn.Layer):
    def __init__(self,num_classes = 30):
        super().__init__()
        self.backbone = STGCN_BackBone()
        self.head = STGCNHead(num_classes = num_classes)
    def forward(self,data):
        feature = self.backbone(data)
        cls_score = self.head(feature)
        return cls_score