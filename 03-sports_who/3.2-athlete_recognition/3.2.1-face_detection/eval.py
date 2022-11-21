from __future__ import absolute_import
from __future__ import division
import yaml
import paddle
import argparse
from utils.logger import setup_logger
from core.trainer import *

logger = setup_logger('eval')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")

    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")

    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")

    parser.add_argument(
        '--save_proposals',
        action='store_true',
        default=False,
        help='Whether to save the train proposals')

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
        '-w',
        '--weights',
        type=str,
        default='None',
        help='model weights')

    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default="configs/blazeface_1000e.yml",
        help='config')

    args = parser.parse_args()
    return args


def load_config(file_path):
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    return file_cfg


def run(args, cfg):

    # build trainer
    trainer = Trainer(cfg, mode='eval')

    # load weights
    if args.weights != 'None':
        trainer.load_weights(args.weights)
    else:
        trainer.load_weights(cfg['weights'])

    trainer.evaluate()


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
