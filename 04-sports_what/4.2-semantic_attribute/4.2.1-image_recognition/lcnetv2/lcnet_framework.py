import paddle.nn as nn
from lcnet_backbone import PPLCNetV2_backbone
from lcnet_head import Classifer_Net
import paddle

class PPLCNetV2_model(nn.Layer):
    def __init__(self,class_num,use_pretrained = "PPLCNetV2_base_pretrained_without_fc.pdparams"):
        super().__init__()
        self.backbone = PPLCNetV2_backbone(scale=1.0, depths=[2, 2, 6, 2],dropout_prob=0.2)
        if use_pretrained != "":
            self.backbone.set_state_dict(paddle.load(use_pretrained))
        self.head = Classifer_Net(1280,class_num)
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x