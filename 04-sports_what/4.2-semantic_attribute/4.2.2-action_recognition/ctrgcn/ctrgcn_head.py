# 导包
import paddle
import paddle.nn as nn
from weight_init import weight_init_
class CTRGCNHead(nn.Layer):
    """
    Head for CTR-GCN model.
    Args:
        in_channels: int, input feature channels. Default: 64.
        num_classes: int, output the number of classes.
        drop_out: float, dropout ratio of layer. Default: 0.
    """

    def __init__(self, in_channels=64, num_classes=30, drop_out=0, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.drop_out = drop_out
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels * 4, self.num_classes)
        if drop_out:
            self.drop_out = nn.Dropout(self.drop_out)
        else:
            self.drop_out = lambda x: x
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters.
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                weight_init_(layer.weight,
                             'Normal',
                             mean=0.0,
                             std=math.sqrt(2. / self.num_classes))

    def forward(self, output_patch):
        """Define how the head is going to run.
        """
        x, N, M = output_patch
        # N*M,C,T,V
        _, c_new, T, V = x.shape
        x = paddle.reshape(x, shape=[N, M, c_new, T * V])
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)