from __future__ import absolute_import
from __future__ import division
import yaml
import paddle
import argparse,os
import glob
from utils.logger import setup_logger
from core.trainer import *

logger = setup_logger('eval')


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")

    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")

    # TODO: bias should be unified
    parser.add_argument(
        "--bias",
        action="store_true",
        help="whether add bias or not while getting w and h")

    parser.add_argument(
        "--classwise",
        action="store_true",
        help="whether per-category AP and draw P-R Curve or not.")

    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')

    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")

    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")

    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")

    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")

    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")

    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default="config/picodet_m_320_coco_lcnet.yml",
        help='config')

    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        default='None',
        help='model weights')

    args = parser.parse_args()
    return args


def load_config(file_path):
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    return file_cfg


def run(args, cfg):

    # build trainer
    trainer = Trainer(cfg, mode='test')

    # load weights
    if args.weights != 'None':
        trainer.load_weights(args.weights)
    else:
        trainer.load_weights(cfg['weights'])

    images = get_test_images(args.infer_dir, args.infer_img)

    trainer.predict(
        images,
        draw_threshold=args.draw_threshold,
        output_dir=args.output_dir,
        save_results=args.save_results,
    )


def print_cfg(cfg,rank):
    if isinstance(cfg,dict):
        for key,value in cfg.items():
            if isinstance(value, dict) or isinstance(value,list):
                print('{}{}:'.format(' '*rank,key))
                print_cfg(value,rank+len(key))
            else:
                print('{}{}:{}'.format(' '*rank,key,value))
    if isinstance(cfg,list):
        for value in cfg:
            if isinstance(value,dict):
                print_cfg(value, rank)
            else:
                print('{}{}'.format(' ' * rank, cfg))
            break


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if cfg['use_gpu']:
        cfg['place'] = paddle.set_device('gpu')
    else:
        cfg['place'] = paddle.set_device('cpu')
    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg['use_gpu']:
        cfg['norm_type'] = 'bn'

    # print_cfg(cfg,0)

    run(args, cfg)


if __name__ == "__main__":
    main()
