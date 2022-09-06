import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np 
from scipy import interpolate

from extractor import *
from update import *
from corr import *
from utils import *


class RAFT(nn.Layer):
    def __init__(self):
        super(RAFT, self).__init__()
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch')
        self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius,hidden_dim=hdim)

    def initialize_flow(self, img):
        # Flow is represented as difference between two coordinate grids flow = coords1 - coords0
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8)
        coords1 = coords_grid(N, H//8, W//8)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
        
    def upsample_flow(self, flow, mask):
        # Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination
        N, _, H, W = flow.shape
        mask = paddle.reshape(mask, shape=[N, 1, 9, 8, 8, H, W])
        mask = F.softmax(mask, axis=2)

        up_flow = F.unfold(8 * flow, kernel_sizes=[3,3], paddings=1)
        up_flow = paddle.reshape(up_flow, shape=[N, 2, 9, 1, 1, H, W])

        up_flow = paddle.sum(mask * up_flow, axis=2)
        up_flow = paddle.transpose(up_flow, perm=[0, 1, 4, 2, 5, 3])

        return paddle.reshape(up_flow, shape=[N, 2, 8*H, 8*W])

    def forward(self, image1, image2, iters=12, flow_init=None, upsample_flow=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # 让元素居于-1到1之间
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # 让tensor连续存储
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        # autocast 是pytorch的混合精度训练模式
        fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        cnet = self.cnet(image1)
        net, inp = paddle.split(cnet, [hdim, cdim], axis=1)
        net = F.tanh(net)
        inp = F.relu(inp)
        
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        
        flow_predictions = []
        
        for itr in range(iters):
            # 首先将coords1从计算图中剥离出来，不会进行梯度下降
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
        
            coords1 = coords1 + delta_flow

            # upsample predictions
            # if up_mask is None: # 我觉得前面的这个不会被执行到，所以插了个桩
            #     flow_up = upflow8(coords1 - coords0)
            #     print("啦啦啦德玛西亚")
            # else:
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)
        
        if test_mode:
            return coords1 - coords0, flow_up
        
        return flow_predictions