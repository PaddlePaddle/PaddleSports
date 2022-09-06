# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from modeling.shape_spec import ShapeSpec
from modeling.backbones.lcnet import DepthwiseSeparable

__all__ = ['LCPAN']


class DPModule(nn.Layer):
    """
    Depth-wise and point-wise module.
     Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_size (int): The conv2d kernel size of this Module.
        stride (int): The conv2d's stride of this Module.
        act (str): The activation function of this Module,
                   Now support `leaky_relu` and `hard_swish`.
    """

    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1,
                 act='leaky_relu',
                 use_act_in_out=True):
        super(DPModule, self).__init__()
        initializer = nn.initializer.KaimingUniform()
        self.use_act_in_out = use_act_in_out
        self.dwconv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channel)
        self.pwconv = nn.Conv2D(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channel)
        if act == "hard_swish":
            act = 'hardswish'
        self.act = act

    def forward(self, x):
        x = self.bn1(self.dwconv(x))
        if self.act:
            x = getattr(F, self.act)(x)
        x = self.bn2(self.pwconv(x))
        if self.use_act_in_out and self.act:
            x = getattr(F, self.act)(x)
        return x


class Channel_T(nn.Layer):
    def __init__(self,
                 in_channels=[116, 232, 464],
                 out_channels=96,
                 act="leaky_relu"):
        super(Channel_T, self).__init__()
        self.convs = nn.LayerList()
        for i in range(len(in_channels)):
            self.convs.append(
                ConvBNLayer(
                    in_channels[i], out_channels, 1, act=act))

    def forward(self, x):
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 act='leaky_relu'):
        super(ConvBNLayer, self).__init__()
        initializer = nn.initializer.KaimingUniform()
        self.conv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channel)
        if act == "hard_swish":
            act = 'hardswish'
        self.act = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act:
            x = getattr(F, self.act)(x)
        return x


class LCPAN(nn.Layer):
    """Path Aggregation Network with LCNet module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_features (int): Number of output features of CSPPAN module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 num_features=3,
                 use_depthwise=True,
                 act='hard_swish',
                 spatial_scales=[0.125, 0.0625, 0.03125]):
        super(LCPAN, self).__init__()
        self.conv_t = Channel_T(in_channels, out_channels, act=act)
        in_channels = [out_channels] * len(spatial_scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        conv_func = DPModule if use_depthwise else ConvBNLayer

        NET_CONFIG = {
            #k, in_c, out_c, stride, use_se
            "block1": [
                [kernel_size, out_channels * 2, out_channels * 2, 1, False],
                [kernel_size, out_channels * 2, out_channels, 1, False],
            ],
            "block2": [
                [kernel_size, out_channels * 2, out_channels * 2, 1, False],
                [kernel_size, out_channels * 2, out_channels, 1, False],
            ]
        }

        if self.num_features == 4:
            self.first_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act)
            self.second_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act)
            self.spatial_scales.append(self.spatial_scales[-1] / 2)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                nn.Sequential(* [
                    DepthwiseSeparable(
                        num_channels=in_c,
                        num_filters=out_c,
                        dw_size=k,
                        stride=s,
                        use_se=se)
                    for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG[
                        "block1"])
                ]))

        # build bottom-up blocks
        self.downsamples = nn.LayerList()
        self.bottom_up_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    in_channels[idx],
                    in_channels[idx],
                    kernel_size=kernel_size,
                    stride=2,
                    act=act))
            self.bottom_up_blocks.append(
                nn.Sequential(* [
                    DepthwiseSeparable(
                        num_channels=in_c,
                        num_filters=out_c,
                        dw_size=k,
                        stride=s,
                        use_se=se)
                    for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG[
                        "block2"])
                ]))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], 1))
            outs.append(out)

        top_features = None
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)

        return tuple(outs)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channels, stride=1. / s)
            for s in self.spatial_scales
        ]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }
