import paddle
from paddle import nn as nn
from paddle.nn import functional as F


# def make_layer(basic_block, num_basic_block, **kwarg):
#     """Make layers by stacking the same blocks.
#
#     Args:
#         basic_block (nn.module): nn.module class for basic block.
#         num_basic_block (int): number of blocks.
#
#     Returns:
#         nn.Sequential: Stacked blocks in nn.Sequential.
#     """
#     layers=[]#List[nn.Layer]
#     for _ in range(num_basic_block):
#         layers.append(basic_block(**kwarg))
#     return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.shape
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.reshape([b, c, h, scale, w, scale])
    return x_view.transpose([0, 1, 3, 5, 2, 4]).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Layer):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2D(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2D(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2D(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2D(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2D(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(paddle.concat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(paddle.concat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(paddle.concat((x, x1, x2, x3), 1)))
        x5 = self.conv5(paddle.concat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Layer):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out1 = self.rdb1(x)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out3 * 0.2 + x


# @ARCH_REGISTRY.register()
class RRDBNet(nn.Layer):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2D(num_in_ch, num_feat, 3, 1, 1)
        self.body = self.make_layer(RRDB, num_block, num_feat, num_grow_ch)
        self.conv_body = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2D(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def make_layer(self, basic_block, num_basic_block, num_feat, num_grow_ch):
        layers = []  # List[nn.Layer]
        for _ in range(num_basic_block):
            layers.append(basic_block(num_feat, num_grow_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        # for mm in self.body:
        #     mm.training=self.conv_first.training
        #     feat = mm(feat)
        body_feat = self.body(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
