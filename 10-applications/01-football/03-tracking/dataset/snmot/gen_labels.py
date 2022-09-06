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

import os.path as osp
import os
import numpy as np

MOT_data = 'SNMOT'

# choose a data in ['MOT15', 'MOT16', 'MOT17', 'MOT20']
# or your custom data (prepare it following the 'docs/tutorials/PrepareMOTDataSet.md')

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_roots = ['./{}/images/train'.format(MOT_data),
        './{}/images/test'.format(MOT_data)]
label_roots = ['./{}/labels_with_ids/train'.format(MOT_data),
        './{}/labels_with_ids/test'.format(MOT_data)]

for seq_root, label_root in zip(seq_roots, label_roots):
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1

    output = {}

    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find(
            '\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find(
            '\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _, _ in gt:
            # import ipdb; ipdb.set_trace()
            if mark == 0 or not label == -1.0:
                continue
            # import ipdb; ipdb.set_trace()
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width,
                h / seq_height)
            if not label_fpath in output:
                output[label_fpath] = []
            output[label_fpath].append(label_str)

    for key in output:
        with open(key, 'w') as f:
            lines = output[key]
            for line in lines:
                f.write(line)
