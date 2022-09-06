import sys
sys.path.append('.')
import os
from data.generate_lq_data import LQDataGenerator
import argparse
import yaml
from collections import OrderedDict



def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

parser = argparse.ArgumentParser()
parser.add_argument('--yml_path', type=str, default='options/train_realesrgan_x4plus.yml', help='yml file')
parser.add_argument('--save_dir', type=str, default='dataset/LQ', help='save folder')
args = parser.parse_args()

def make_lq_data(yml_path, save_dir):
        # parse yml to dict
    with open(yml_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    
    lqDataGenerator = LQDataGenerator(opt)
    lqDataGenerator.generate_lq_data(save_dir)


    
if __name__ == "__main__":
    yml_path = args.yml_path
    save_dir = args.save_dir
    make_lq_data(yml_path, save_dir)
