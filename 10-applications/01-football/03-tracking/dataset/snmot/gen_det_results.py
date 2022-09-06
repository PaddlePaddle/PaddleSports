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

def mkdirs_safe(d):
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)

seq_roots = ['./{}/images/train'.format(MOT_data),
        './{}/images/test'.format(MOT_data),
        './{}/images/challenge'.format(MOT_data)]
det_roots = ['./{}/det_files/train'.format(MOT_data),
        './{}/det_files/test'.format(MOT_data),
        './{}/det_files/challenge'.format(MOT_data)]

for seq_root, det_root in zip(seq_roots, det_roots):
    mkdirs_safe(det_root)
    seqs = [s for s in os.listdir(seq_root)]
    for seq in seqs:
        print(seq)
        src_det_txt = osp.join(seq_root, seq, 'det', 'det.txt')
        src_det = np.loadtxt(src_det_txt, dtype=np.float64, delimiter=',')
        #only extract 7 items per row: [frame_id],[x0],[y0],[w],[h],[score],[class_id]
        dst_det = src_det[:, [0,2,3,4,5,6,7]]
        dst_det_txt = osp.join(det_root, f'{seq}.txt')
        np.savetxt(dst_det_txt, dst_det, fmt='%d', delimiter=',')

