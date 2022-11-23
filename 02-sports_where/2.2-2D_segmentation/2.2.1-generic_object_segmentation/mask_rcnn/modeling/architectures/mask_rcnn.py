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
from modeling.proposal_generator.rpn_head import RPNHead
from modeling.heads.bbox_head import BBoxHead
from modeling.backbones.resnet import Res5Head
from modeling.proposal_generator.target_layer import BBoxAssigner,MaskAssigner
from modeling.heads.mask_head import MaskFeat,MaskHead
from modeling.post_process import BBoxPostProcess,MaskPostProcess
from modeling.layers import MultiClassNMS,RCNNBox

__all__ = ['MaskRCNN']


class MaskRCNN(nn.Layer):
    """
    Mask R-CNN network, see https://arxiv.org/abs/1703.06870

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        mask_head (object): `MaskHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        mask_post_process (object): `MaskPostProcess` instance
        neck (object): 'FPN' instance
    """

    __category__ = 'architecture'
    __inject__ = [
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 cfg,
                 backbone,
                 rpn_head,
                 bbox_head,
                 mask_head,
                 bbox_post_process,
                 mask_post_process,
                 data_format='NCHW',
                 neck=None):
        super(MaskRCNN, self).__init__()
        self.data_format = data_format
        self.inputs = {}
        self.fuse_norm = False
        self.cfg = cfg

        # build backbone
        for backbone_cls, args in backbone.items():
            for k, v in args.items():
                # build object args
                if isinstance(v, dict) and v.get('name', False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    args[k] = globals()[obj_cls](**_args)

            self.backbone = globals()[backbone_cls](**args)

        # build rpn_head
        in_channel = [_.channels for _ in self.backbone.out_shape]
        for rpn_head_cls, args in rpn_head.items():
            for k, v in args.items():
                # build object args
                if isinstance(v, dict) and v.get('name', False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    args[k] = globals()[obj_cls](**_args)
            self.rpn_head = globals()[rpn_head_cls]( **args,in_channel=in_channel[0])

        # build bbox_head
        for bbox_head_cls, args in bbox_head.items():
            for k, v in args.items():
                # build object args
                if isinstance(v, dict) and v.get('name', False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    args[k] = globals()[obj_cls](**_args)
            self.bbox_head = globals()[bbox_head_cls](**args,in_channel=in_channel[0]*2)

        # build mask_head
        for mask_head_cls,args in mask_head.items():
            for k, v in args.items():
                # build object args
                if isinstance(v, dict) and v.get('name', False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    if obj_cls == 'MaskFeat':
                        args[k] = globals()[obj_cls](**_args,in_channel=in_channel[0]*2)
                    else:
                        args[k] = globals()[obj_cls](**_args)
            self.mask_head = globals()[mask_head_cls](**args)

        # build bbox_post_process
        for bbox_post_process_cls,args in bbox_post_process.items():
            for k, v in args.items():
                # build object args
                if isinstance(v, dict) and v.get('name', False):
                    _args = v
                    obj_cls = _args['name']
                    del _args['name']
                    args[k] = globals()[obj_cls](**_args)
            self.bbox_post_process = globals()[bbox_post_process_cls](**args)

        # creat mask_post_process
        for mask_post_process_cls,args in mask_post_process.items():
            self.mask_post_process = globals()[mask_post_process_cls](**args)

        self.neck = neck


    def from_config(cls, cfg, *args, **kwargs):
        pass

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)
            bbox_loss, bbox_feat = self.bbox_head(body_feats, rois, rois_num,self.inputs)
            rois, rois_num = self.bbox_head.get_assigned_rois()
            bbox_targets = self.bbox_head.get_assigned_targets()
            # Mask Head needs bbox_feat in Mask RCNN
            mask_loss = self.mask_head(body_feats, rois, rois_num, self.inputs,
                                       bbox_targets, bbox_feat)
            return rpn_loss, bbox_loss, mask_loss
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            preds, feat_func = self.bbox_head(body_feats, rois, rois_num, None)

            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']

            bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
                                                    im_shape, scale_factor)
            mask_out = self.mask_head(
                body_feats, bbox, bbox_num, self.inputs, feat_func=feat_func)

            # rescale the prediction back to origin image
            bbox, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
                bbox, bbox_num, im_shape, scale_factor)
            origin_shape = self.bbox_post_process.get_origin_shape()
            mask_pred = self.mask_post_process(mask_out, bbox_pred, bbox_num,
                                               origin_shape)
            return bbox_pred, bbox_num, mask_pred

    def get_loss(self, ):
        bbox_loss, mask_loss, rpn_loss = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        loss.update(mask_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num, mask_pred = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num, 'mask': mask_pred}
        return output

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