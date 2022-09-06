# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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
from modeling.backbones.lcnet import *
from modeling.necks.lc_pan import *
from modeling.heads.pico_head import *
from modeling.assigners.atss_assigner import *
from modeling.assigners.task_aligned_assigner import *
from modeling.losses.varifocal_loss import *
from modeling.losses.gfocal_loss import *
from modeling.layers import MultiClassNMS
from modeling.post_process import nms
from modeling.losses.iou_loss import *

__all__ = ['PicoDet']


class PicoDet(nn.Layer):
    """
    Generalized Focal Loss network, see https://arxiv.org/abs/2006.04388

    Args:
        backbone (object): backbones instance
        neck (object): 'FPN' instance
        head (object): 'PicoHead' instance
    """

    # __category__ = 'architecture'xxx

    def __init__(self,cfg,backbone, neck, head='PicoHead',data_format='NCHW'):
        super(PicoDet, self).__init__()
        self.data_format = data_format
        self.inputs = {}
        self.fuse_norm = False
        self.cfg = cfg

        # build backbone
        for backbone_cls,args in backbone.items():
            self.backbone = globals()[backbone_cls](**args)

        # build neck
        in_channels = [_.channels for _ in self.backbone.out_shape]
        for neck_cls, args in neck.items():
            self.neck = globals()[neck_cls](in_channels,**args)

        # build head
        # kwargs = {'input_shape': self.neck.out_shape}
        for head_cls, args in head.items():
            for k,v in args.items():
                # build object args
                if isinstance(v,dict) and v.get('name',False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    args[k] = globals()[obj_cls](**_args)

            self.head = globals()[head_cls](**args)

        self.export_post_process = True
        self.export_nms = True

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

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        head_outs = self.head(fpn_feats, self.export_post_process)
        if self.training or not self.export_post_process:
            return head_outs, None
        else:
            scale_factor = self.inputs['scale_factor']
            bboxes, bbox_num = self.head.post_process(
                head_outs, scale_factor, export_nms=self.export_nms)
            return bboxes, bbox_num

    def get_loss(self, ):
        loss = {}

        head_outs, _ = self._forward()
        loss_gfl = self.head.get_loss(head_outs, self.inputs)
        loss.update(loss_gfl)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        if not self.export_post_process:
            return {'picodet': self._forward()[0]}
        elif self.export_nms:
            bbox_pred, bbox_num = self._forward()
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
            return output
        else:
            bboxes, mlvl_scores = self._forward()
            output = {'bbox': bboxes, 'scores': mlvl_scores}
            return output
