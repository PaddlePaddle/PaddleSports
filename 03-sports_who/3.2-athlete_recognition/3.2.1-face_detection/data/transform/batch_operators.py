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

import typing

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import cv2
import math
import numpy as np
from .operators import BaseOperator, Resize
from .op_helper import jaccard_overlap, gaussian2D, gaussian_radius, draw_umich_gaussian
from scipy import ndimage
import data.transform.bbox_utils
from utils.logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    'BatchRandomResize',
    'PadGT',
    'PadBatch'
]


class BatchRandomResize(BaseOperator):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio,
                 interp=cv2.INTER_NEAREST,
                 random_size=True,
                 random_interp=False):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.interp = interp
        assert isinstance(target_size, (
            int, Sequence)), "target_size must be int, list or tuple"
        if random_size and not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List, now is {}".
                    format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples, context=None):
        if self.random_size:
            index = np.random.choice(len(self.target_size))
            target_size = self.target_size[index]
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, keep_ratio=self.keep_ratio, interp=interp)
        return resizer(samples, context=context)


class PadGT(BaseOperator):
    """
    Pad 0 to `gt_class`, `gt_bbox`, `gt_score`...
    The num_max_boxes is the largest for batch.
    Args:
        return_gt_mask (bool): If true, return `pad_gt_mask`,
                                1 means bbox, 0 means no bbox.
    """

    def __init__(self, return_gt_mask=True):
        super(PadGT, self).__init__()
        self.return_gt_mask = return_gt_mask

    def __call__(self, samples, context=None):
        num_max_boxes = max([len(s['gt_bbox']) for s in samples])
        for sample in samples:
            if self.return_gt_mask:
                sample['pad_gt_mask'] = np.zeros(
                    (num_max_boxes, 1), dtype=np.float32)
            if num_max_boxes == 0:
                continue

            num_gt = len(sample['gt_bbox'])
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt] = sample['gt_class']
                pad_gt_bbox[:num_gt] = sample['gt_bbox']
            sample['gt_class'] = pad_gt_class
            sample['gt_bbox'] = pad_gt_bbox
            # pad_gt_mask
            if 'pad_gt_mask' in sample:
                sample['pad_gt_mask'][:num_gt] = 1
            # gt_score
            if 'gt_score' in sample:
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_score[:num_gt] = sample['gt_score']
                sample['gt_score'] = pad_gt_score
            if 'is_crowd' in sample:
                pad_is_crowd = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_is_crowd[:num_gt] = sample['is_crowd']
                sample['is_crowd'] = pad_is_crowd
            if 'difficult' in sample:
                pad_diff = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_diff[:num_gt] = sample['difficult']
                sample['difficult'] = pad_diff
        return samples


class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        # multi scale input is nested list
        if isinstance(samples,
                      typing.Sequence) and len(samples) > 0 and isinstance(
                          samples[0], typing.Sequence):
            inner_samples = samples[0]
        else:
            inner_samples = samples

        max_shape = np.array(
            [data['image'].shape for data in inner_samples]).max(axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in inner_samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm

            if 'gt_rbox2poly' in data and data['gt_rbox2poly'] is not None:
                # ploy to rbox
                polys = data['gt_rbox2poly']
                rbox = bbox_utils.poly2rbox(polys)
                data['gt_rbox'] = rbox

        return samples
