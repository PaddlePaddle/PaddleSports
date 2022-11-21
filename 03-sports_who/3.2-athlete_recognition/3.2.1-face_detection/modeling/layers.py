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
import six
import math
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


class SSDBox(object):
    def __init__(self,
                 is_normalized=True,
                 prior_box_var=[0.1, 0.1, 0.2, 0.2],
                 use_fuse_decode=False):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)
        self.prior_box_var = prior_box_var
        self.use_fuse_decode = use_fuse_decode

    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes, scores = preds
        boxes = paddle.concat(boxes, axis=1)
        prior_boxes = paddle.concat(prior_boxes)
        if self.use_fuse_decode:
            output_boxes = ops.box_coder(
                prior_boxes,
                self.prior_box_var,
                boxes,
                code_type="decode_center_size",
                box_normalized=self.is_normalized)
        else:
            pb_w = prior_boxes[:, 2] - prior_boxes[:, 0] + self.norm_delta
            pb_h = prior_boxes[:, 3] - prior_boxes[:, 1] + self.norm_delta
            pb_x = prior_boxes[:, 0] + pb_w * 0.5
            pb_y = prior_boxes[:, 1] + pb_h * 0.5
            out_x = pb_x + boxes[:, :, 0] * pb_w * self.prior_box_var[0]
            out_y = pb_y + boxes[:, :, 1] * pb_h * self.prior_box_var[1]
            out_w = paddle.exp(boxes[:, :, 2] * self.prior_box_var[2]) * pb_w
            out_h = paddle.exp(boxes[:, :, 3] * self.prior_box_var[3]) * pb_h
            output_boxes = paddle.stack(
                [
                    out_x - out_w / 2., out_y - out_h / 2., out_x + out_w / 2.,
                    out_y + out_h / 2.
                ],
                axis=-1)

        if self.is_normalized:
            h = (im_shape[:, 0] / scale_factor[:, 0]).unsqueeze(-1)
            w = (im_shape[:, 1] / scale_factor[:, 1]).unsqueeze(-1)
            im_shape = paddle.stack([w, h, w, h], axis=-1)
            output_boxes *= im_shape
        else:
            output_boxes[..., -2:] -= 1.0
        output_scores = F.softmax(paddle.concat(
            scores, axis=1)).transpose([0, 2, 1])

        return output_boxes, output_scores


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


class MaskMatrixNMS(object):
    """
    Matrix NMS for multi-class masks.
    Args:
        update_threshold (float): Updated threshold of categroy score in second time.
        pre_nms_top_n (int): Number of total instance to be kept per image before NMS
        post_nms_top_n (int): Number of total instance to be kept per image after NMS.
        kernel (str):  'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
    Input:
        seg_preds (Variable): shape (n, h, w), segmentation feature maps
        seg_masks (Variable): shape (n, h, w), segmentation feature maps
        cate_labels (Variable): shape (n), mask labels in descending order
        cate_scores (Variable): shape (n), mask scores in descending order
        sum_masks (Variable): a float tensor of the sum of seg_masks
    Returns:
        Variable: cate_scores, tensors of shape (n)
    """

    def __init__(self,
                 update_threshold=0.05,
                 pre_nms_top_n=500,
                 post_nms_top_n=100,
                 kernel='gaussian',
                 sigma=2.0):
        super(MaskMatrixNMS, self).__init__()
        self.update_threshold = update_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.kernel = kernel
        self.sigma = sigma

    def _sort_score(self, scores, top_num):
        if paddle.shape(scores)[0] > top_num:
            return paddle.topk(scores, top_num)[1]
        else:
            return paddle.argsort(scores, descending=True)

    def __call__(self,
                 seg_preds,
                 seg_masks,
                 cate_labels,
                 cate_scores,
                 sum_masks=None):
        # sort and keep top nms_pre
        sort_inds = self._sort_score(cate_scores, self.pre_nms_top_n)
        seg_masks = paddle.gather(seg_masks, index=sort_inds)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        sum_masks = paddle.gather(sum_masks, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)

        seg_masks = paddle.flatten(seg_masks, start_axis=1, stop_axis=-1)
        # inter.
        inter_matrix = paddle.mm(seg_masks, paddle.transpose(seg_masks, [1, 0]))
        n_samples = paddle.shape(cate_labels)
        # union.
        sum_masks_x = paddle.expand(sum_masks, shape=[n_samples, n_samples])
        # iou.
        iou_matrix = (inter_matrix / (
            sum_masks_x + paddle.transpose(sum_masks_x, [1, 0]) - inter_matrix))
        iou_matrix = paddle.triu(iou_matrix, diagonal=1)
        # label_specific matrix.
        cate_labels_x = paddle.expand(cate_labels, shape=[n_samples, n_samples])
        label_matrix = paddle.cast(
            (cate_labels_x == paddle.transpose(cate_labels_x, [1, 0])),
            'float32')
        label_matrix = paddle.triu(label_matrix, diagonal=1)

        # IoU compensation
        compensate_iou = paddle.max((iou_matrix * label_matrix), axis=0)
        compensate_iou = paddle.expand(
            compensate_iou, shape=[n_samples, n_samples])
        compensate_iou = paddle.transpose(compensate_iou, [1, 0])

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'gaussian':
            decay_matrix = paddle.exp(-1 * self.sigma * (decay_iou**2))
            compensate_matrix = paddle.exp(-1 * self.sigma *
                                           (compensate_iou**2))
            decay_coefficient = paddle.min(decay_matrix / compensate_matrix,
                                           axis=0)
        elif self.kernel == 'linear':
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient = paddle.min(decay_matrix, axis=0)
        else:
            raise NotImplementedError

        # update the score.
        cate_scores = cate_scores * decay_coefficient
        y = paddle.zeros(shape=paddle.shape(cate_scores), dtype='float32')
        keep = paddle.where(cate_scores >= self.update_threshold, cate_scores,
                            y)
        keep = paddle.nonzero(keep)
        keep = paddle.squeeze(keep, axis=[1])
        # Prevent empty and increase fake data
        keep = paddle.concat(
            [keep, paddle.cast(paddle.shape(cate_scores)[0] - 1, 'int64')])

        seg_preds = paddle.gather(seg_preds, index=keep)
        cate_scores = paddle.gather(cate_scores, index=keep)
        cate_labels = paddle.gather(cate_labels, index=keep)

        # sort and keep top_k
        sort_inds = self._sort_score(cate_scores, self.post_nms_top_n)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)
        return seg_preds, cate_scores, cate_labels


class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=[8, 16, 32, 64, 100, 300],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
                 min_ratio=15,
                 max_ratio=90,
                 base_size=300,
                 min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
                 max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
                 offset=0.5,
                 flip=True,
                 clip=False,
                 min_max_aspect_ratios_order=False):
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_size = base_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order

        if self.min_sizes == [] and self.max_sizes == []:
            num_layer = len(aspect_ratios)
            step = int(
                math.floor(((self.max_ratio - self.min_ratio)) / (num_layer - 2
                                                                  )))
            for ratio in six.moves.range(self.min_ratio, self.max_ratio + 1,
                                         step):
                self.min_sizes.append(self.base_size * ratio / 100.)
                self.max_sizes.append(self.base_size * (ratio + step) / 100.)
            self.min_sizes = [self.base_size * .10] + self.min_sizes
            self.max_sizes = [self.base_size * .20] + self.max_sizes

        self.num_priors = []
        for aspect_ratio, min_size, max_size in zip(
                aspect_ratios, self.min_sizes, self.max_sizes):
            if isinstance(min_size, (list, tuple)):
                self.num_priors.append(
                    len(_to_list(min_size)) + len(_to_list(max_size)))
            else:
                self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(
                    _to_list(min_size)) + len(_to_list(max_size)))

    def __call__(self, inputs, image):
        boxes = []
        for input, min_size, max_size, aspect_ratio, step in zip(
                inputs, self.min_sizes, self.max_sizes, self.aspect_ratios,
                self.steps):
            box, _ = ops.prior_box(
                input=input,
                image=image,
                min_sizes=_to_list(min_size),
                max_sizes=_to_list(max_size),
                aspect_ratios=aspect_ratio,
                flip=self.flip,
                clip=self.clip,
                steps=[step, step],
                offset=self.offset,
                min_max_aspect_ratios_order=self.min_max_aspect_ratios_order)
            boxes.append(paddle.reshape(box, [-1, 4]))
        return boxes

