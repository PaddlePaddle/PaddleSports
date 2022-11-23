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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from modeling.bbox_utils import nonempty_bbox

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


class JDEBBoxPostProcess(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['decode', 'nms']

    def __init__(self, num_classes=1, decode=None, nms=None, return_idx=True):
        super(JDEBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.return_idx = return_idx

        self.fake_bbox_pred = paddle.to_tensor(
            np.array(
                [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
        self.fake_bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))
        self.fake_nms_keep_idx = paddle.to_tensor(
            np.array(
                [[0]], dtype='int32'))

        self.fake_yolo_boxes_out = paddle.to_tensor(
            np.array(
                [[[0.0, 0.0, 0.0, 0.0]]], dtype='float32'))
        self.fake_yolo_scores_out = paddle.to_tensor(
            np.array(
                [[[0.0]]], dtype='float32'))
        self.fake_boxes_idx = paddle.to_tensor(np.array([[0]], dtype='int64'))

    def forward(self, head_out, anchors):
        """
        Decode the bbox and do NMS for JDE model.

        Args:
            head_out (list): Bbox_pred and cls_prob of bbox_head output.
            anchors (list): Anchors of JDE model.

        Returns:
            boxes_idx (Tensor): The index of kept bboxes after decode 'JDEBox'.
            bbox_pred (Tensor): The output is the prediction with shape [N, 6]
                including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction of each batch with shape [N].
            nms_keep_idx (Tensor): The index of kept bboxes after NMS.
        """
        boxes_idx, yolo_boxes_scores = self.decode(head_out, anchors)

        if len(boxes_idx) == 0:
            boxes_idx = self.fake_boxes_idx
            yolo_boxes_out = self.fake_yolo_boxes_out
            yolo_scores_out = self.fake_yolo_scores_out
        else:
            yolo_boxes = paddle.gather_nd(yolo_boxes_scores, boxes_idx)
            # TODO: only support bs=1 now
            yolo_boxes_out = paddle.reshape(
                yolo_boxes[:, :4], shape=[1, len(boxes_idx), 4])
            yolo_scores_out = paddle.reshape(
                yolo_boxes[:, 4:5], shape=[1, 1, len(boxes_idx)])
            boxes_idx = boxes_idx[:, 1:]

        if self.return_idx:
            bbox_pred, bbox_num, nms_keep_idx = self.nms(
                yolo_boxes_out, yolo_scores_out, self.num_classes)
            if bbox_pred.shape[0] == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
                nms_keep_idx = self.fake_nms_keep_idx
            return boxes_idx, bbox_pred, bbox_num, nms_keep_idx
        else:
            bbox_pred, bbox_num, _ = self.nms(yolo_boxes_out, yolo_scores_out,
                                              self.num_classes)
            if bbox_pred.shape[0] == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
            return _, bbox_pred, bbox_num, _


class BBoxPostProcess(object):
    __shared__ = ['num_classes', 'export_onnx']
    __inject__ = ['decode', 'nms']

    def __init__(self, num_classes=80, decode=None, nms=None,
                 export_onnx=False):
        super(BBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.export_onnx = export_onnx

    def __call__(self, head_out, rois, im_shape, scale_factor):
        """
        Decode the bbox and do NMS if needed.

        Args:
            head_out (tuple): bbox_pred and cls_prob of bbox_head output.
            rois (tuple): roi and rois_num of rpn_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
            export_onnx (bool): whether export model to onnx
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
        """
        if self.nms is not None:
            bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
            bbox_pred, bbox_num, _ = self.nms(bboxes, score, self.num_classes)

        else:
            bbox_pred, bbox_num = self.decode(head_out, rois, im_shape,
                                              scale_factor)

        if self.export_onnx:
            # add fake box after postprocess when exporting onnx
            fake_bboxes = paddle.to_tensor(
                np.array(
                    [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype='float32'))

            bbox_pred = paddle.concat([bbox_pred, fake_bboxes])
            bbox_num = bbox_num + 1

        return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to
        get final prediction.

        Notes:
        Currently only support bs = 1.

        Args:
            bboxes (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            pred_result (Tensor): The final prediction results with shape [N, 6]
                including labels, scores and bboxes.
        """
        if not self.export_onnx:
            bboxes_list = []
            bbox_num_list = []
            id_start = 0
            fake_bboxes = paddle.to_tensor(
                np.array(
                    [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype='float32'))
            fake_bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))

            # add fake bbox when output is empty for each batch
            for i in range(bbox_num.shape[0]):
                if bbox_num[i] == 0:
                    bboxes_i = fake_bboxes
                    bbox_num_i = fake_bbox_num
                else:
                    bboxes_i = bboxes[id_start:id_start + bbox_num[i], :]
                    bbox_num_i = bbox_num[i]
                    id_start += bbox_num[i]
                bboxes_list.append(bboxes_i)
                bbox_num_list.append(bbox_num_i)
            bboxes = paddle.concat(bboxes_list)
            bbox_num = paddle.concat(bbox_num_list)

        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

        if not self.export_onnx:
            origin_shape_list = []
            scale_factor_list = []
            # scale_factor: scale_y, scale_x
            for i in range(bbox_num.shape[0]):
                expand_shape = paddle.expand(origin_shape[i:i + 1, :],
                                             [bbox_num[i], 2])
                scale_y, scale_x = scale_factor[i][0], scale_factor[i][1]
                scale = paddle.concat([scale_x, scale_y, scale_x, scale_y])
                expand_scale = paddle.expand(scale, [bbox_num[i], 4])
                origin_shape_list.append(expand_shape)
                scale_factor_list.append(expand_scale)

            self.origin_shape_list = paddle.concat(origin_shape_list)
            scale_factor_list = paddle.concat(scale_factor_list)

        else:
            # simplify the computation for bs=1 when exporting onnx
            scale_y, scale_x = scale_factor[0][0], scale_factor[0][1]
            scale = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y]).unsqueeze(0)
            self.origin_shape_list = paddle.expand(origin_shape,
                                                   [bbox_num[0], 2])
            scale_factor_list = paddle.expand(scale, [bbox_num[0], 4])

        # bboxes: [N, 6], label, score, bbox
        pred_label = bboxes[:, 0:1]
        pred_score = bboxes[:, 1:2]
        pred_bbox = bboxes[:, 2:]
        # rescale bbox to original image
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = self.origin_shape_list[:, 0]
        origin_w = self.origin_shape_list[:, 1]
        zeros = paddle.zeros_like(origin_h)
        # clip bbox to [0, original_size]
        x1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 3], origin_h), zeros)
        pred_bbox = paddle.stack([x1, y1, x2, y2], axis=-1)
        # filter empty bbox
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = paddle.unsqueeze(keep_mask, [1])
        pred_label = paddle.where(keep_mask, pred_label,
                                  paddle.ones_like(pred_label) * -1)
        pred_result = paddle.concat([pred_label, pred_score, pred_bbox], axis=1)
        return bboxes, pred_result, bbox_num

    def get_origin_shape(self, ):
        return self.origin_shape_list


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    # _i, _j
    # sorted indices
    # i, j
    # temp variables for box i's (the box currently under consideration)
    # ix1, iy1, ix2, iy2, iarea

    # variables for computing overlap with box j (lower scoring box)
    # xx1, yy1, xx2, yy2
    # w, h
    # inter, ovr

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets


class MaskPostProcess(object):
    __shared__ = ['export_onnx', 'assign_on_cpu']
    """
    refer to:
    https://github.com/facebookresearch/detectron2/layers/mask_ops.py

    Get Mask output according to the output from model
    """

    def __init__(self,
                 binary_thresh=0.5,
                 export_onnx=False,
                 assign_on_cpu=False):
        super(MaskPostProcess, self).__init__()
        self.binary_thresh = binary_thresh
        self.export_onnx = export_onnx
        self.assign_on_cpu = assign_on_cpu

    def paste_mask(self, masks, boxes, im_h, im_w):
        """
        Paste the mask prediction to the original image.
        """
        x0_int, y0_int = 0, 0
        x1_int, y1_int = im_w, im_h
        x0, y0, x1, y1 = paddle.split(boxes, 4, axis=1)
        N = masks.shape[0]
        img_y = paddle.arange(y0_int, y1_int) + 0.5
        img_x = paddle.arange(x0_int, x1_int) + 0.5

        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        # img_x, img_y have shapes (N, w), (N, h)

        if self.assign_on_cpu:
            paddle.set_device('cpu')
        gx = img_x[:, None, :].expand(
            [N, paddle.shape(img_y)[1], paddle.shape(img_x)[1]])
        gy = img_y[:, :, None].expand(
            [N, paddle.shape(img_y)[1], paddle.shape(img_x)[1]])
        grid = paddle.stack([gx, gy], axis=3)
        img_masks = F.grid_sample(masks, grid, align_corners=False)
        return img_masks[:, 0]

    def __call__(self, mask_out, bboxes, bbox_num, origin_shape):
        """
        Decode the mask_out and paste the mask to the origin image.

        Args:
            mask_out (Tensor): mask_head output with shape [N, 28, 28].
            bbox_pred (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            origin_shape (Tensor): The origin shape of the input image, the tensor
                shape is [N, 2], and each row is [h, w].
        Returns:
            pred_result (Tensor): The final prediction mask results with shape
                [N, h, w] in binary mask style.
        """
        num_mask = mask_out.shape[0]
        origin_shape = paddle.cast(origin_shape, 'int32')
        device = paddle.device.get_device()

        if self.export_onnx:
            h, w = origin_shape[0][0], origin_shape[0][1]
            mask_onnx = self.paste_mask(mask_out[:, None, :, :], bboxes[:, 2:],
                                        h, w)
            mask_onnx = mask_onnx >= self.binary_thresh
            pred_result = paddle.cast(mask_onnx, 'int32')

        else:
            max_h = paddle.max(origin_shape[:, 0])
            max_w = paddle.max(origin_shape[:, 1])
            pred_result = paddle.zeros(
                [num_mask, max_h, max_w], dtype='int32') - 1

            id_start = 0
            for i in range(paddle.shape(bbox_num)[0]):
                bboxes_i = bboxes[id_start:id_start + bbox_num[i], :]
                mask_out_i = mask_out[id_start:id_start + bbox_num[i], :, :]
                im_h = origin_shape[i, 0]
                im_w = origin_shape[i, 1]
                bbox_num_i = bbox_num[id_start]
                pred_mask = self.paste_mask(mask_out_i[:, None, :, :],
                                            bboxes_i[:, 2:], im_h, im_w)
                pred_mask = paddle.cast(pred_mask >= self.binary_thresh,
                                        'int32')
                pred_result[id_start:id_start + bbox_num[i], :im_h, :
                            im_w] = pred_mask
                id_start += bbox_num[i]
        if self.assign_on_cpu:
            paddle.set_device(device)

        return pred_result