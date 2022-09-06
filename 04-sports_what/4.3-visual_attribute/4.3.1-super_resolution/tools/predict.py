import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse
import cv2
import glob
import numpy as np
import paddle

from models.rrdbnet_arch import RRDBNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./inputs/00003.png', help='Input image or folder')
    parser.add_argument(
        '--model_path',
        type=str,
        # default = '/home/aistudio/work/Real-ESRGAN-paddle1129/experiments/pretrained_models/RealESRGAN_x4plus.pdparams',
        default='./experiments/pretrained_models_1/net_g_latest7.pdparams',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    # parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    # parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    # parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    # parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument('--block', type=int, default=9, help='num_block in RRDB')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_known_args()[0]

    if 'RealESRGAN_x4plus_anime_6B.pdparams' in args.model_path:
        args.block = 6
    elif 'RealESRGAN_x2plus.pdparams' in args.model_path:
        args.netscale = 2

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=args.block, num_grow_ch=32, scale=args.netscale)
    loadnet = paddle.load(args.model_path)
    model.set_state_dict(loadnet["params"])
    model.eval()

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
        img = paddle.Tensor(np.transpose(img, (2, 0, 1)))
        img = img.unsqueeze(0)

        # ------------------- process image (without the alpha channel) ------------------- #
        with paddle.no_grad():
            output = model(img)
        output_img = output.squeeze().numpy().clip(0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            alpha = alpha.astype(np.float32)
            alpha = paddle.Tensor(np.transpose(alpha, (2, 0, 1)))
            out_alpha = alpha.unsqueeze(0)

            with paddle.no_grad():
                output_alpha = model(out_alpha)
            output_alpha = paddle.clip(output_alpha.squeeze().astype(paddle.float32).cpu(), 0, 1).numpy()
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


if __name__ == '__main__':
    main()