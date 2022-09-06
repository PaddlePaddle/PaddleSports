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
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config

######

import numpy as np
import paddle
from paddle.io import Dataset
import os
import cv2
import json

from ppcls.data import preprocess
from ppcls.data.preprocess import transform
from ppcls.utils import logger
from ppcls.data.dataloader.common_dataset import create_operators
from PIL import Image



class SNReIDDataset(Dataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 data_type=None,
                 is_challenge=None,
                 transform_ops=None,
                 backend="cv2"):
        self._img_root = image_root
        self._cls_path = os.path.join(image_root, cls_label_path)
        self._dataset_dir = image_root
        self._transform_ops = transform_ops
        self._data_type = data_type
        self._check_before_run()
        self.images = []
        self.labels = []
        self.cameras = []

        if not is_challenge:
            with open(self._cls_path, 'r', encoding='UTF-8') as f:
                self._bbox_infos = json.load(f)
            if data_type in ['query', 'gallery']:
                self._bbox_infos = self._bbox_infos[data_type]
            self._bbox_infos = list(self._bbox_infos.values())
            for info in self._bbox_infos:
                bbox_idx = info['bbox_idx']
                action_idx = info['action_idx']
                person_uid = info['person_uid']
                frame_idx = info['frame_idx']
                clazz = info['clazz']
                ID = info['id']
                UAI = info['UAI']
                height = info['height']
                width = info['width']
                relative_path = info['relative_path']
                if data_type in ['query', 'gallery']:
                    relative_path = os.path.join(data_type, relative_path)
                img_path = os.path.join(image_root, relative_path, f'{bbox_idx}-{action_idx}-{person_uid}-{frame_idx}-{clazz}-{ID}-{UAI}-{height}x{width}.png')
                if not os.path.exists(img_path):
                    continue
                self.images.append(img_path)
                self.labels.append(person_uid)
                self.cameras.append(action_idx)
        else:
            img_dir = os.path.join(image_root, data_type)
            imgs = os.listdir(img_dir)
            img_idxs = [[int(_item) for _item in item.split('-')] for item in imgs]
            img_paths = [os.path.join(image_root, data_type, item) for item in imgs]
            for img_path, [bbox_idx, action_idx] in zip(img_paths, img_idxs):
                if not os.path.exists(img_path):
                    continue
                self.images.append(img_path)
                self.labels.append(bbox_idx)
                self.cameras.append(action_idx)
                
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        self.backend = backend
        self._dtype = paddle.get_default_dtype()

    def _check_before_run(self):
        """Check if the file is available before going deeper"""
        if not os.path.exists(self._dataset_dir):
            raise RuntimeError("'{}' is not available".format(
                self._dataset_dir))

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.backend == "cv2":
                img = np.array(img, dtype="float32").astype(np.uint8)
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            if self.backend == "cv2":
                img = img.transpose((2, 0, 1))
            return (img, self.labels[idx])
        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(img_path, ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels)) 


######
import inspect
import copy
import paddle
import numpy as np
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader
from ppcls.utils import logger

from ppcls.data import dataloader
# dataset
from ppcls.data.dataloader.imagenet_dataset import ImageNetDataset
from ppcls.data.dataloader.multilabel_dataset import MultiLabelDataset
from ppcls.data.dataloader.common_dataset import create_operators
from ppcls.data.dataloader.vehicle_dataset import CompCars, VeriWild
from ppcls.data.dataloader.logo_dataset import LogoDataset
from ppcls.data.dataloader.icartoon_dataset import ICartoonDataset
from ppcls.data.dataloader.mix_dataset import MixDataset
from ppcls.data.dataloader.multi_scale_dataset import MultiScaleDataset
from ppcls.data.dataloader.person_dataset import Market1501, MSMT17
from ppcls.data.dataloader.face_dataset import FiveValidationDataset, AdaFaceDataset


# sampler
from ppcls.data.dataloader.DistributedRandomIdentitySampler import DistributedRandomIdentitySampler
from ppcls.data.dataloader.pk_sampler import PKSampler
from ppcls.data.dataloader.mix_sampler import MixSampler
from ppcls.data.dataloader.multi_scale_sampler import MultiScaleSampler
from ppcls.data import preprocess
from ppcls.data.preprocess import transform


def build_dataloader(config, mode, device, use_dali=False, seed=None):
    assert mode in [
        'Train', 'Eval', 'Test', 'Gallery', 'Query'
    ], "Dataset mode should be Train, Eval, Test, Gallery, Query"
    # build dataset
    if use_dali:
        from ppcls.data.dataloader.dali import dali_dataloader
        return dali_dataloader(config, mode, paddle.device.get_device(), seed)

    class_num = config.get("class_num", None)
    config_dataset = config[mode]['dataset']
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')
    if 'batch_transform_ops' in config_dataset:
        batch_transform = config_dataset.pop('batch_transform_ops')
    else:
        batch_transform = None

    dataset = SNReIDDataset(**config_dataset)

    logger.debug("build dataset({}) success...".format(dataset))

    # build sampler
    config_sampler = config[mode]['sampler']
    if config_sampler and "name" not in config_sampler:
        batch_sampler = None
        batch_size = config_sampler["batch_size"]
        drop_last = config_sampler["drop_last"]
        shuffle = config_sampler["shuffle"]
    else:
        sampler_name = config_sampler.pop("name")
        batch_sampler = eval(sampler_name)(dataset, **config_sampler)

    logger.debug("build batch_sampler({}) success...".format(batch_sampler))

    # build batch operator
    def mix_collate_fn(batch):
        batch = transform(batch, batch_ops)
        # batch each field
        slots = []
        for items in batch:
            for i, item in enumerate(items):
                if len(slots) < len(items):
                    slots.append([item])
                else:
                    slots[i].append(item)
        return [np.stack(slot, axis=0) for slot in slots]

    if isinstance(batch_transform, list):
        batch_ops = create_operators(batch_transform, class_num)
        batch_collate_fn = mix_collate_fn
    else:
        batch_collate_fn = None

    # build dataloader
    config_loader = config[mode]['loader']
    num_workers = config_loader["num_workers"]
    use_shared_memory = config_loader["use_shared_memory"]

    if batch_sampler is None:
        data_loader = DataLoader(
            dataset=dataset,
            places=device,
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=batch_collate_fn)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            places=device,
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_sampler=batch_sampler,
            collate_fn=batch_collate_fn)

    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader

###### fix build_dataloader ######
import ppcls.data
ppcls.data.build_dataloader = build_dataloader

######
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    config.profiler_options = args.profiler_options
    engine = Engine(config, mode="train")
    engine.train()
