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
            return (img, self.labels[idx], self.cameras[idx])
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

import platform
from typing import Optional

import numpy as np
import paddle
from ppcls.utils import logger
from ppcls.engine.evaluation.retrieval import re_ranking, cal_feature

def get_ranking_results(distmat, q_pids, q_action_indices, g_action_indices, export=False):
    q_pids, q_action_indices, g_action_indices = q_pids.reshape((-1,)), q_action_indices.reshape((-1,)), g_action_indices.reshape((-1,))
    num_q, _ = distmat.shape
    indices = paddle.argsort(distmat, axis=1)
    num_valid_q = 0
    ranking_results = {}
    for q_idx in range(num_q):
        # get query pid and action_idx
        q_action_idx = q_action_indices[q_idx]

        # remove gallery samples from different action than the query
        order = indices[q_idx]
        remove = (g_action_indices[order] != q_action_idx)
        keep = paddle.bitwise_not(remove)
        g_ranking = order[keep]

        if g_ranking.size == 0:
            print("Does not appear in gallery: q_idx {} - q_pid {} - q_action_idx {}".format(q_idx, q_pids[q_idx], q_action_idx))
            # this condition is true when query identity does not appear in gallery
            continue

        ranking_results[str(q_idx)] = g_ranking.tolist()
        num_valid_q += 1.
    if export:
        with open(export, 'w', encoding='UTF-8') as f:
            json.dump(ranking_results, f)
    return ranking_results


def snreid_evaluate(grountruth, ranking_results):
    queries_gt = grountruth["query"]
    galleries_gt = grountruth["gallery"]

    # action2gidx dict maps each action_idx to a list of gallery_idx from that action
    action2gidx = {}
    for gallery in galleries_gt.values():
        action_idx = gallery["action_idx"]
        if action_idx not in action2gidx:
            action2gidx[action_idx] = []
        action2gidx[action_idx].append(gallery["bbox_idx"])

    all_cmc = []
    all_AP = []
    max_rank = 1

    for q_idx in queries_gt.keys():
        query_gt = queries_gt[q_idx]
        q_pid = query_gt["person_uid"]
        q_action_idx = query_gt["action_idx"]

        if q_idx not in ranking_results:
            raise ValueError("No ranking provided for query '{}'. Make sure to provide ranking result for all queries.".format(q_idx))

        seen_galleries = set()
        gallery_ranking_idx = ranking_results[q_idx]
        gallery_ranking_pid = []
        for g_idx in gallery_ranking_idx:
            if not isinstance(g_idx, int):
                raise TypeError("Incorrect ranking result for query '{}'. Ranking must contain integer values only but contained '{}' of type '{}'".format(q_idx, g_idx, type(g_idx)))
            if g_idx in seen_galleries:
                raise ValueError("Gallery sample '{}' is referenced more than once in ranking result of query '{}'. "
                                 "Each gallery sample from the same action as the query must appear no more and not less than once in the ranking.".format(g_idx, q_idx))
            else:
                seen_galleries.add(g_idx)
            gallery_gt = galleries_gt[str(g_idx)]
            g_action_idx = gallery_gt["action_idx"]
            if q_action_idx != g_action_idx:
                raise ValueError("Ranking result for query '{}' from action '{}' contained gallery sample '{}' from a different action '{}'. "
                                 "Ranking results for a given query must contain only gallery samples from the same action.".format(q_idx, q_action_idx, g_idx, g_action_idx))
            gallery_ranking_pid.append(gallery_gt["person_uid"])

        # make sure all gallery samples are provided for given query
        gallery_ranking_idx_set = set(gallery_ranking_idx)
        for g_idx in action2gidx[q_action_idx]:
            if g_idx not in gallery_ranking_idx_set:
                raise ValueError("Ranking result for query '{}' is incorrect: missing gallery sample '{}' from same action '{}'. "
                                 "A gallery sample from the same action as the query should be listed in the ranking.".format(q_idx, g_idx, q_action_idx))

        matches = []
        for g_pid in gallery_ranking_pid:
            matches.append(g_pid == q_pid)

        raw_cmc = np.array(matches)
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            raise ValueError("Person id for query '{}' does not appear in gallery".format(q_idx))

        cmc1 = raw_cmc.cumsum()
        cmc1[cmc1 > 1] = 1
        cmc1 = cmc1[:max_rank]
        all_cmc.append(cmc1)

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i1 + 1.) for i1, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / len(queries_gt.keys())
    ap = np.mean(all_AP)
    result = all_cmc, ap
    cmc, mAP = result

    performance_metrics = {}
    performance_metrics["mAP"] = mAP
    performance_metrics["rank-1"] = cmc[0]

    return performance_metrics

def retrieval_eval(engine, epoch_id=0):
    engine.model.eval()
    # step1. build gallery
    if engine.gallery_query_dataloader is not None:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery_query')
        query_feas, query_img_id, query_query_id = gallery_feas, gallery_img_id, gallery_unique_id
    else:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery')
        query_feas, query_img_id, query_query_id = cal_feature(
            engine, name='query')

    # step2. do evaluation
    sim_block_size = engine.config["Global"].get("sim_block_size", 64)
    sections = [sim_block_size] * (len(query_feas) // sim_block_size)
    if len(query_feas) % sim_block_size:
        sections.append(len(query_feas) % sim_block_size)
    fea_blocks = paddle.split(query_feas, num_or_sections=sections)
    if query_query_id is not None:
        query_id_blocks = paddle.split(
            query_query_id, num_or_sections=sections)
    image_id_blocks = paddle.split(query_img_id, num_or_sections=sections)
    metric_key = None
    metric_dict = dict()

    # compute distance matrix(The smaller the value, the more similar)
    distmat = re_ranking(
        query_feas, gallery_feas, k1=20, k2=6, lambda_value=0.3)

    snreid_export_flag = engine.config['Global'].get('snreid_export', False)
    image_root = engine.config['DataLoader']['Eval']['Query']['dataset']['image_root']
    cls_label_path = engine.config['DataLoader']['Eval']['Query']['dataset']['cls_label_path']
    ranking_results = get_ranking_results(distmat, query_img_id, query_query_id, gallery_unique_id, snreid_export_flag)
    gt_json_path = os.path.join(image_root, cls_label_path)
    with open(gt_json_path, 'r', encoding='UTF-8') as f:
        grountruth = json.load(f)
    metric_dict = snreid_evaluate(grountruth, ranking_results)

    metric_info_list = []
    for key in metric_dict:
        if metric_key is None:
            metric_key = key
        metric_info_list.append("{}: {:.5f}".format(key, metric_dict[key]))
    metric_msg = ", ".join(metric_info_list)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    return metric_dict[metric_key]

######
import ppcls.engine.evaluation as evaluation
evaluation.retrieval_eval = retrieval_eval

######
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    engine = Engine(config, mode="eval")
    engine.eval()
