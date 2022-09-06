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
import cv2
import numpy as np
import paddle
from paddle import inference
import glob

def load_predictor(model_file_path, params_file_path):
    """load_predictor
    initialize the inference engine
    Args:
        model_file_path: inference model path (*.pdmodel)
        model_file_path: inference parmaeter path (*.pdiparams)
    Return:
        predictor: Predictor created using Paddle Inference.
        input_tensor: Input tensor of the predictor.
        output_tensor: Output tensor of the predictor.
    """
    config = inference.Config(model_file_path, params_file_path)
    config.enable_use_gpu(1000, 0)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)

    # get input and output tensor property
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    return predictor, input_tensor, output_tensor

def str2bool(v):
    return v.lower() in ("true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        '--model_path', default='./infer/', help='path where to save')
    parser.add_argument('--input', type=str, default='./inputs/00003.png', help='Input image or folder')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_known_args()[0]

    predictor, input_tensor, output_tensor = load_predictor(
        os.path.join(args.model_path, "inference.pdmodel"),
        os.path.join(args.model_path, "inference.pdiparams"))

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range

        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        out_img = np.expand_dims(img, axis=0)
        # pre_pad
        # out_img = F.pad(out_img, (0, 10, 0, 10), 'reflect')

        input_tensor.copy_from_cpu(out_img)
        predictor.run()
        output = output_tensor.copy_to_cpu()

        output_img = output.squeeze().clip(0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            alpha = alpha.astype(np.float32)
            alpha = np.transpose(alpha, (2, 0, 1))
            out_alpha = np.expand_dims(alpha, axis=0)

            input_tensor.copy_from_cpu(out_alpha)
            predictor.run()
            output_alpha = output_tensor.copy_to_cpu()

            output_alpha = output_alpha.squeeze().clip(0, 1)
            output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output_img = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output_img = (output_img * 255.0).round().astype(np.uint8)

        output_img = cv2.resize(
            output_img, (
                int(w_input * args.outscale),
                int(h_input * args.outscale),
            ), interpolation=cv2.INTER_LANCZOS4)

        if args.ext == 'auto':
            extension = extension[1:]
        else:
            extension = args.ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        cv2.imwrite(save_path, output_img)



