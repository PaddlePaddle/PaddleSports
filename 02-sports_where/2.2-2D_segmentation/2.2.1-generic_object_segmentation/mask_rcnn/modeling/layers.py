#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant
from paddle.regularizer import L2Decay
from modeling import ops
from paddle.vision.ops import DeformConv2D
from modeling.bbox_utils import delta2bbox


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class DropBlock(nn.Layer):
    def __init__(self, block_size, keep_prob, name=None, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = paddle.cast(paddle.rand(x.shape) < gamma, x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=True,
                 nms_eta=1.0,
                 return_index=False,
                 return_rois_num=True,
                 trt=False):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,]
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1.
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        if background_label > -1:
            kwargs.update({'background_label': background_label})
        kwargs.pop('trt')
        # TODO(wangxinxin08): paddle version should be develop or 2.3 and above to run nms on tensorrt
        if self.trt and (int(paddle.version.major) == 0 or
                         (int(paddle.version.major) >= 2 and
                          int(paddle.version.minor) >= 3)):
            # TODO(wangxinxin08): tricky switch to run nms on tensorrt
            kwargs.update({'nms_eta': 1.1})
            bbox, bbox_num, _ = ops.multiclass_nms(bboxes, score, **kwargs)
            mask = paddle.slice(bbox, [-1], [0], [1]) != -1
            bbox = paddle.masked_select(bbox, mask).reshape((-1, 6))
            return bbox, bbox_num, None
        else:
            return ops.multiclass_nms(bboxes, score, **kwargs)


class RCNNBox(object):
    __shared__ = ['num_classes', 'export_onnx']

    def __init__(self,
                 prior_box_var=[10., 10., 5., 5.],
                 code_type="decode_center_size",
                 box_normalized=False,
                 num_classes=80,
                 export_onnx=False):
        super(RCNNBox, self).__init__()
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.num_classes = num_classes
        self.export_onnx = export_onnx

    def __call__(self, bbox_head_out, rois, im_shape, scale_factor):
        bbox_pred = bbox_head_out[0]
        cls_prob = bbox_head_out[1]
        roi = rois[0]
        rois_num = rois[1]

        if self.export_onnx:
            onnx_rois_num_per_im = rois_num[0]
            origin_shape = paddle.expand(im_shape[0, :],
                                         [onnx_rois_num_per_im, 2])

        else:
            origin_shape_list = []
            if isinstance(roi, list):
                batch_size = len(roi)
            else:
                batch_size = paddle.slice(paddle.shape(im_shape), [0], [0], [1])

            # bbox_pred.shape: [N, C*4]
            for idx in range(batch_size):
                rois_num_per_im = rois_num[idx]
                expand_im_shape = paddle.expand(im_shape[idx, :],
                                                [rois_num_per_im, 2])
                origin_shape_list.append(expand_im_shape)

            origin_shape = paddle.concat(origin_shape_list)

        # bbox_pred.shape: [N, C*4]
        # C=num_classes in faster/mask rcnn(bbox_head), C=1 in cascade rcnn(cascade_head)
        bbox = paddle.concat(roi)
        bbox = delta2bbox(bbox_pred, bbox, self.prior_box_var)
        scores = cls_prob[:, :-1]

        # bbox.shape: [N, C, 4]
        # bbox.shape[1] must be equal to scores.shape[1]
        total_num = bbox.shape[0]
        bbox_dim = bbox.shape[-1]
        bbox = paddle.expand(bbox, [total_num, self.num_classes, bbox_dim])

        origin_h = paddle.unsqueeze(origin_shape[:, 0], axis=1)
        origin_w = paddle.unsqueeze(origin_shape[:, 1], axis=1)
        zeros = paddle.zeros_like(origin_h)
        x1 = paddle.maximum(paddle.minimum(bbox[:, :, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(bbox[:, :, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(bbox[:, :, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(bbox[:, :, 3], origin_h), zeros)
        bbox = paddle.stack([x1, y1, x2, y2], axis=-1)
        bboxes = (bbox, rois_num)
        return bboxes, scores


class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 skip_quant=False,
                 dcn_bias_regularizer=L2Decay(0.),
                 dcn_bias_lr_scale=2.):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = ParamAttr(initializer=Constant(0.))
        else:
            offset_bias_attr = ParamAttr(
                initializer=Constant(0.),
                learning_rate=lr_scale,
                regularizer=regularizer)
        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=offset_bias_attr)
        if skip_quant:
            self.conv_offset.skip_quant = True

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = ParamAttr(
                initializer=Constant(value=0),
                regularizer=dcn_bias_regularizer,
                learning_rate=dcn_bias_lr_scale)
        else:
            # in ResNet backbones, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=dcn_bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=Normal(
                     mean=0., std=0.01),
                 skip_quant=False,
                 dcn_lr_scale=2.,
                 dcn_regularizer=L2Decay(0.)):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]

        if bias_on:
            bias_attr = ParamAttr(
                initializer=Constant(value=0.), learning_rate=lr_scale)
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=bias_attr)
            if skip_quant:
                self.conv.skip_quant = True
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=True,
                lr_scale=dcn_lr_scale,
                regularizer=dcn_regularizer,
                dcn_bias_regularizer=dcn_regularizer,
                dcn_bias_lr_scale=dcn_lr_scale,
                skip_quant=skip_quant)

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2D(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)
        else:
            self.norm = None

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)
        return out
