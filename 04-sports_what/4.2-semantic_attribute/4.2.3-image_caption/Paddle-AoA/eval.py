from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import misc.eval_utils as eval_utils
import argparse
import misc.utils as utils
import modules.losses as loss
import paddle

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# config
parser.add_argument('--cfg', type=str, default='configs/eval.yml',
                    help='configuration; similar to what is used in detectron')
opts.add_eval_options(parser)

opt = parser.parse_args()

# read cfg_fn
from yacs.config import CfgNode as CN
if opt.cfg is not None:
    fcfg = open(opt.cfg, 'rb')
    cfg = CN.load_cfg(fcfg)
    for k, v in cfg.items():
        if not hasattr(opt, k):
            print('Warning: key %s not in args' % k)
        setattr(opt, k, v)
    args = parser.parse_args(namespace=opt)

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_dict(paddle.load(opt.model))

use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device('gpu:0')

model.eval()
crit = loss.LanguageModelCriterion()

# Create the Data Loader instance
loader = DataLoader(opt)

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# Set sample options
opt.dataset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
