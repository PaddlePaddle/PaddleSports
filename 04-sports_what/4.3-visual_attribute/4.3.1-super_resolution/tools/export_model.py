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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse
import paddle
from paddle.static import InputSpec
from models.rrdbnet_arch import RRDBNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_inference_dir', default='./infer/', help='path where to save')
    parser.add_argument('--input', type=str, default='./inputs/00003.png', help='Input image or folder')
    parser.add_argument(
        'model_path',
        type=str,
        default = './experiments/pretrained_models_1/net_g_latest7.pdparams',
        help='Path to the pre-trained model')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--block', type=int, default=9, help='num_block in RRDB')
    args = parser.parse_known_args()[0]

    test_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=args.block, num_grow_ch=32, scale=args.netscale)
    loadnet = paddle.load(args.model_path)
    test_model.set_state_dict(loadnet["params"])
    test_model.eval()

    # decorate model with jit.save
    model = paddle.jit.to_static(
        test_model,
        input_spec=[
            InputSpec(shape=[-1, 3, -1, -1], dtype='float32', name='x')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")
