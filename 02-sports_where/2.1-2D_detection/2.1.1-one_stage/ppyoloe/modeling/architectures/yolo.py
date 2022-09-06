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
from modeling.post_process import JDEBBoxPostProcess,BBoxPostProcess
from modeling.backbones.cspresnet import CSPResNet
from modeling.necks.custom_pan import CustomCSPPAN
from modeling.heads.ppyoloe_head import PPYOLOEHead
from modeling.assigners.atss_assigner import *
from modeling.assigners.task_aligned_assigner import *
from modeling.post_process import nms
from modeling.layers import MultiClassNMS

__all__ = ['YOLOv3']


class YOLOv3(nn.Layer):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 cfg,
                 backbone='DarkNet',
                 neck='YOLOv3FPN',
                 yolo_head='YOLOv3Head',
                 post_process='BBoxPostProcess',
                 data_format='NCHW',
                 for_mot=False):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            bbox_post_process (object): `BBoxPostProcess` instance
            data_format (str): data format, NCHW or NHWC
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv3, self).__init__()
        self.data_format = data_format
        self.inputs = {}
        self.fuse_norm = False
        self.cfg = cfg

        # build backbone
        for backbone_cls, args in backbone.items():
            self.backbone = globals()[backbone_cls](**args)

        # build neck
        in_channels = [_.channels for _ in self.backbone.out_shape]
        for neck_cls, args in neck.items():
            self.neck = globals()[neck_cls](in_channels, **args)

        # build head
        # kwargs = {'input_shape': self.neck.out_shape}
        for head_cls, args in yolo_head.items():
            for k, v in args.items():
                # build object args
                if isinstance(v, dict) and v.get('name', False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    args[k] = globals()[obj_cls](**_args)

            self.yolo_head = globals()[head_cls](**args)

        # build post_process
        self.post_process = post_process

        self.for_mot = for_mot
        self.return_idx = isinstance(post_process, JDEBBoxPostProcess)

    def from_config(cls, cfg, *args, **kwargs):
        pass

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if isinstance(neck_feats, dict):
            assert self.for_mot == True
            emb_feats = neck_feats['emb_feats']
            neck_feats = neck_feats['yolo_feats']

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)

            if self.for_mot:
                return {'det_losses': yolo_losses, 'emb_feats': emb_feats}
            else:
                return yolo_losses

        else:
            yolo_head_outs = self.yolo_head(neck_feats)

            if self.for_mot:
                boxes_idx, bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors)
                output = {
                    'bbox': bbox,
                    'bbox_num': bbox_num,
                    'boxes_idx': boxes_idx,
                    'nms_keep_idx': nms_keep_idx,
                    'emb_feats': emb_feats,
                }
            else:
                if self.return_idx:
                    _, bbox, bbox_num, _ = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors)
                elif self.post_process is not None:
                    bbox, bbox_num = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors,
                        self.inputs['im_shape'], self.inputs['scale_factor'])
                else:
                    bbox, bbox_num = self.yolo_head.post_process(
                        yolo_head_outs, self.inputs['im_shape'],
                        self.inputs['scale_factor'])
                output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

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

    def model_arch(self, ):
        pass

