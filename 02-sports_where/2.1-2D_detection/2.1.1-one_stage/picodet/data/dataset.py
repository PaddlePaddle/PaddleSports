# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import os
import numpy as np

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from paddle.io import Dataset
import copy
import cv2
from data.crop_utils.annotation_cropper import AnnoCropper
from utils.logger import setup_logger
logger = setup_logger('reader')


class DetDataset(Dataset):
    """
    Load detection datasets.

    Args:
        dataset_dir (str): root directory for datasets.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        data_fields (list): key name of datasets dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether to load default label list.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.use_default_label = use_default_label
        self._epoch = 0
        self._curr_iter = 0

    def __len__(self, ):
        return len(self.roidbs)

    def __getitem__(self, idx):
        # datasets batch
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            n = len(self.roidbs)
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(4)
            ]
        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
        else:
            roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1
        return self.transform(roidb)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self, ):
        raise NotImplementedError(
            "Need to implement parse_dataset method of Dataset")

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)


class COCODataSet(DetDataset):
    """
    Load datasets with COCO format.

    Args:
        dataset_dir (str): root directory for datasets.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of datasets dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total
            record's, if empty_ratio is out of [0. ,1.), do not sample the
            records and use all the empty entries. 1. as default
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.):
        super(COCODataSet, self).__init__(dataset_dir, image_dir, anno_path,
                                          data_fields, sample_num)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    # read rbox anno or not
                    is_rbox_anno = True if len(inst['bbox']) == 5 else False
                    if is_rbox_anno:
                        xc, yc, box_w, box_h, angle = inst['bbox']
                        x1 = xc - box_w / 2.0
                        y1 = yc - box_h / 2.0
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                    else:
                        x1, y1, box_w, box_h = inst['bbox']
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        if is_rbox_anno:
                            inst['clean_rbox'] = [xc, yc, box_w, box_h, angle]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                if is_rbox_anno:
                    gt_rbox = np.zeros((num_bbox, 5), dtype=np.float32)
                gt_theta = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    # xc, yc, w, h, theta
                    if is_rbox_anno:
                        gt_rbox[i, :] = box['clean_rbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(box['segmentation']
                                        ).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    continue

                if is_rbox_anno:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_rbox': gt_rbox,
                        'gt_poly': gt_poly,
                    }
                else:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_poly': gt_poly,
                    }

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records


class SniperCOCODataSet(COCODataSet):
    """SniperCOCODataSet"""

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 proposals_file=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=True,
                 empty_ratio=1.,
                 is_trainset=True,
                 image_target_sizes=[2000, 1000],
                 valid_box_ratio_ranges=[[-1, 0.1],[0.08, -1]],
                 chip_target_size=500,
                 chip_target_stride=200,
                 use_neg_chip=False,
                 max_neg_num_per_im=8,
                 max_per_img=-1,
                 nms_thresh=0.5):
        super(SniperCOCODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            load_crowd=load_crowd,
            allow_empty=allow_empty,
            empty_ratio=empty_ratio
        )
        self.proposals_file = proposals_file
        self.proposals = None
        self.anno_cropper = None
        self.is_trainset = is_trainset
        self.image_target_sizes = image_target_sizes
        self.valid_box_ratio_ranges = valid_box_ratio_ranges
        self.chip_target_size = chip_target_size
        self.chip_target_stride = chip_target_stride
        self.use_neg_chip = use_neg_chip
        self.max_neg_num_per_im = max_neg_num_per_im
        self.max_per_img = max_per_img
        self.nms_thresh = nms_thresh

    def parse_dataset(self):
        if not hasattr(self, "roidbs"):
            super(SniperCOCODataSet, self).parse_dataset()
        if self.is_trainset:
            self._parse_proposals()
            self._merge_anno_proposals()
        self.ori_roidbs = copy.deepcopy(self.roidbs)
        self.init_anno_cropper()
        self.roidbs = self.generate_chips_roidbs(self.roidbs, self.is_trainset)

    def set_proposals_file(self, file_path):
        self.proposals_file = file_path

    def init_anno_cropper(self):
        logger.info("Init AnnoCropper...")
        self.anno_cropper = AnnoCropper(
            image_target_sizes=self.image_target_sizes,
            valid_box_ratio_ranges=self.valid_box_ratio_ranges,
            chip_target_size=self.chip_target_size,
            chip_target_stride=self.chip_target_stride,
            use_neg_chip=self.use_neg_chip,
            max_neg_num_per_im=self.max_neg_num_per_im,
            max_per_img=self.max_per_img,
            nms_thresh=self.nms_thresh
        )

    def generate_chips_roidbs(self, roidbs, is_trainset):
        if is_trainset:
            roidbs = self.anno_cropper.crop_anno_records(roidbs)
        else:
            roidbs = self.anno_cropper.crop_infer_anno_records(roidbs)
        return roidbs

    def _parse_proposals(self):
        if self.proposals_file:
            self.proposals = {}
            logger.info("Parse proposals file:{}".format(self.proposals_file))
            with open(self.proposals_file, 'r') as f:
                proposals = json.load(f)
            for prop in proposals:
                image_id = prop["image_id"]
                if image_id not in self.proposals:
                    self.proposals[image_id] = []
                x, y, w, h = prop["bbox"]
                self.proposals[image_id].append([x, y, x + w, y + h])

    def _merge_anno_proposals(self):
        assert self.roidbs
        if self.proposals and len(self.proposals.keys()) > 0:
            logger.info("merge proposals to annos")
            for id, record in enumerate(self.roidbs):
                image_id = int(record["im_id"])
                if image_id not in self.proposals.keys():
                    logger.info("image id :{} no proposals".format(image_id))
                record["proposals"] = np.array(self.proposals.get(image_id, []), dtype=np.float32)
                self.roidbs[id] = record

    def get_ori_roidbs(self):
        if not hasattr(self, "ori_roidbs"):
            return None
        return self.ori_roidbs

    def get_roidbs(self):
        if not hasattr(self, "roidbs"):
            self.parse_dataset()
        return self.roidbs

    def set_roidbs(self, roidbs):
        self.roidbs = roidbs

    def check_or_download_dataset(self):
        return

    def _parse(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._parse()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            im = cv2.imread(image)
            h, w, c = im.shape
            rec = {'im_id': np.array([ct]), 'im_file': image, "h": h, "w": w}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images):
        self._imid2path = {}
        self.image_dir = images
        self.roidbs = self._load_images()


def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    return f.lower().endswith(extensions)