import paddle
import paddle.nn.functional as F 
from utils import *

# 本文件涉及了计算光流的成本量函数


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = paddle.reshape(corr, shape=[batch*h1*w1, dim, h2, w2])        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = paddle.transpose(coords, perm=[0, 2, 3, 1])
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            # linspace为线性等分 r为搜索半径
            dx = paddle.linspace(-r, r, 2*r+1)
            dy = paddle.linspace(-r, r, 2*r+1)
            delta = paddle.stack(paddle.meshgrid(dy, dx), axis=-1)
            
            centroid_lvl = paddle.reshape(coords, shape=[batch*h1*w1, 1, 1, 2]) / 2**i
            delta_lvl = paddle.reshape(delta, shape=[1, 2*r+1, 2*r+1, 2])
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = paddle.reshape(corr, shape=[batch, h1, w1, -1])
            out_pyramid.append(corr)
        out = paddle.concat(out_pyramid, axis=-1)
        return paddle.transpose(out, perm=[0, 3, 1, 2]).astype('float32')
    # transpose、permute 操作虽然没有修改底层一维数组，但是新建了一份Tensor元信息，并在新的元信息中的重新指定stride。
    # torch.view 方法约定了不修改数组本身，只是使用新的形状查看数据。

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        # 修改tensor的形状
        fmap1 = paddle.reshape(fmap1, shape=[batch, dim, ht*wd])
        fmap2 = paddle.reshape(fmap2, shape=[batch, dim, ht*wd]) 

        corr = paddle.matmul(paddle.transpose(fmap1, perm=[0,2,1]), fmap2)
        corr = paddle.reshape(corr, shape=[batch, ht, wd, 1, ht, wd])
        
        # 归一化？！
        return corr / paddle.sqrt(paddle.to_tensor(float(dim)))


