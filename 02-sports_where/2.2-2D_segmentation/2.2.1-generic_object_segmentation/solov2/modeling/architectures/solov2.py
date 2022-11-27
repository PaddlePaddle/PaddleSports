# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import numpy as np
import typing

from modeling.post_process import nms
from modeling.backbones.resnet import ResNet
from modeling.necks.fpn import FPN
from modeling.heads.solov2_head import SOLOv2Head,SOLOv2MaskHead
from modeling.losses.solov2_loss import SOLOv2Loss
from modeling.layers import MaskMatrixNMS


__all__ = ['SOLOv2']


class SOLOv2(nn.Layer):
    """
    SOLOv2 network, see https://arxiv.org/abs/2003.10152

    Args:
        backbone (object): an backbone instance
        solov2_head (object): an `SOLOv2Head` instance
        mask_head (object): an `SOLOv2MaskHead` instance
        neck (object): neck of network, such as feature pyramid network instance
    """

    __category__ = 'architecture'

    def __init__(self,
                 cfg,
                 backbone,
                 solov2_head,
                 mask_head,
                 data_format='NCHW',
                 neck=None):
        super(SOLOv2, self).__init__()
        self.inputs = {}
        self.cfg = cfg
        self.fuse_norm = False
        self.data_format = data_format

        self.backbone = globals()[list(backbone.keys())[0]](**list(backbone.values())[0])
        in_channel = [_.channels for _ in self.backbone.out_shape]

        self.neck = globals()[list(neck.keys())[0]](**list(neck.values())[0],in_channels=in_channel)

        for solov2_head_cls, args in solov2_head.items():
            if solov2_head_cls == 'SOLOv2Head':
                args['num_classes'] = cfg['num_classes']
            for k, v in args.items():
                # build object args
                if isinstance(v, dict) and v.get('name', False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    args[k] = globals()[obj_cls](**_args)
            self.solov2_head = globals()[solov2_head_cls](**args)

        in_channel = [_.channels for _ in self.neck.out_shape]
        mask_head['in_channels'] = in_channel[0]
        self.mask_head = globals()[list(mask_head.keys())[0]](**list(mask_head.values())[0],in_channels=in_channel[0])

    def model_arch(self):
        body_feats = self.backbone(self.inputs)

        body_feats = self.neck(body_feats)

        self.seg_pred = self.mask_head(body_feats)
        self.cate_pred_list, self.kernel_pred_list = self.solov2_head(
            body_feats)

    def get_loss(self, ):
        loss = {}
        # get gt_ins_labels, gt_cate_labels, etc.
        gt_ins_labels, gt_cate_labels, gt_grid_orders = [], [], []
        fg_num = self.inputs['fg_num']
        for i in range(len(self.solov2_head.seg_num_grids)):
            ins_label = 'ins_label{}'.format(i)
            if ins_label in self.inputs:
                gt_ins_labels.append(self.inputs[ins_label])
            cate_label = 'cate_label{}'.format(i)
            if cate_label in self.inputs:
                gt_cate_labels.append(self.inputs[cate_label])
            grid_order = 'grid_order{}'.format(i)
            if grid_order in self.inputs:
                gt_grid_orders.append(self.inputs[grid_order])

        loss_solov2 = self.solov2_head.get_loss(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            gt_ins_labels, gt_cate_labels, gt_grid_orders, fg_num)
        loss.update(loss_solov2)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        seg_masks, cate_labels, cate_scores, bbox_num = self.solov2_head.get_prediction(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            self.inputs['im_shape'], self.inputs['scale_factor'])
        outs = {
            "segm": seg_masks,
            "bbox_num": bbox_num,
            'cate_label': cate_labels,
            'cate_score': cate_scores
        }
        return outs

    def load_meanstd(self, cfg_transform):
        scale = 1.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for item in cfg_transform:
            if 'NormalizeImage' in item:
                mean = np.array(
                    item['NormalizeImage']['mean'], dtype=np.float32)
                std = np.array(item['NormalizeImage']['std'], dtype=np.float32)
                if item['NormalizeImage'].get('is_scale', True):
                    scale = 1. / 255.
                break
        if self.data_format == 'NHWC':
            self.scale = paddle.to_tensor(scale / std).reshape((1, 1, 1, 3))
            self.bias = paddle.to_tensor(-mean / std).reshape((1, 1, 1, 3))
        else:
            self.scale = paddle.to_tensor(scale / std).reshape((1, 3, 1, 1))
            self.bias = paddle.to_tensor(-mean / std).reshape((1, 3, 1, 1))

    def forward(self, inputs):
        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = paddle.transpose(image, [0, 2, 3, 1])

        if self.fuse_norm:
            image = inputs['image']
            self.inputs['image'] = image * self.scale + self.bias
            self.inputs['im_shape'] = inputs['im_shape']
            self.inputs['scale_factor'] = inputs['scale_factor']
        else:
            self.inputs = inputs

        self.model_arch()

        if self.training:
            out = self.get_loss()
        else:
            inputs_list = []
            # multi-scale input
            if not isinstance(inputs, typing.Sequence):
                inputs_list.append(inputs)
            else:
                inputs_list.extend(inputs)
            outs = []
            for inp in inputs_list:
                if self.fuse_norm:
                    self.inputs['image'] = inp['image'] * self.scale + self.bias
                    self.inputs['im_shape'] = inp['im_shape']
                    self.inputs['scale_factor'] = inp['scale_factor']
                else:
                    self.inputs = inp
                outs.append(self.get_pred())

            # multi-scale test
            if len(outs) > 1:
                out = self.merge_multi_scale_predictions(outs)
            else:
                out = outs[0]
        return out

    def merge_multi_scale_predictions(self, outs):
        # default values for architectures not included in following list
        num_classes = 80
        nms_threshold = 0.5
        keep_top_k = 100

        if self.__class__.__name__ in ('CascadeRCNN', 'FasterRCNN', 'MaskRCNN'):
            num_classes = self.bbox_head.num_classes
            keep_top_k = self.bbox_post_process.nms.keep_top_k
            nms_threshold = self.bbox_post_process.nms.nms_threshold
        else:
            raise Exception(
                "Multi scale test only supports CascadeRCNN, FasterRCNN and MaskRCNN for now"
            )

        final_boxes = []
        all_scale_outs = paddle.concat([o['bbox'] for o in outs]).numpy()
        for c in range(num_classes):
            idxs = all_scale_outs[:, 0] == c
            if np.count_nonzero(idxs) == 0:
                continue
            r = nms(all_scale_outs[idxs, 1:], nms_threshold)
            final_boxes.append(
                np.concatenate([np.full((r.shape[0], 1), c), r], 1))
        out = np.concatenate(final_boxes)
        out = np.concatenate(sorted(
            out, key=lambda e: e[1])[-keep_top_k:]).reshape((-1, 6))
        out = {
            'bbox': paddle.to_tensor(out),
            'bbox_num': paddle.to_tensor(np.array([out.shape[0], ]))
        }

        return out

    def build_inputs(self, data, input_def):
        inputs = {}
        for i, k in enumerate(input_def):
            inputs[k] = data[i]
        return inputs
