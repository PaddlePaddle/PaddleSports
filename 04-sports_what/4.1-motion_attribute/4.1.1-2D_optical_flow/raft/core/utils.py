import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np 
from scipy import interpolate

from extractor import *
from update import *
from corr import *
from utils import *



def coords_grid(batch, ht, wd): # device 原版的参数中还包括device，我觉得没有必要，就取消了，但愿后期不会一半在cpu，一半在GPU
    # 生成网格
    coords = paddle.meshgrid(paddle.arange(ht), paddle.arange(wd))
    # 两个通道对调位置
    coords = paddle.stack(coords[::-1], axis=0)
    return paddle.repeat_interleave(coords[None], repeats=batch, axis=0).astype('float32')


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid =coords.split([1, 1], axis=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = paddle.concat([xgrid, ygrid], axis=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img
