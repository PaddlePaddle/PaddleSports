"""
Modified from https://github.com/mlomnitz/DiffJPEG

For images not divisible by 8
https://dsp.stackexchange.com/questions/35339/jpeg-dct-padding/35343#35343
"""
import itertools
import numpy as np
import paddle
paddle.enable_static()
paddle.disable_static()
import paddle.nn as nn
from paddle.nn import functional as F

# ------------------------ utils ------------------------#
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
y_table = paddle.Tensor(y_table)
# y_table = paddle.create_parameter(y_table.shape, dtype='float32',default_initializer=paddle.nn.initializer.Assign(y_table))
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = paddle.Tensor(c_table)
# c_table = paddle.create_parameter(c_table.shape, dtype='float32', default_initializer=paddle.nn.initializer.Assign(c_table))


def diff_round(x):
    """ Differentiable rounding function
    """
    return paddle.round(x) + (x - paddle.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality

    Args:
        quality(float): Quality for jpeg compression.

    Returns:
        float: Compression factor.
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


# ------------------------ compression ------------------------#
class RGB2YCbCrJpeg(nn.Layer):
    """ Converts RGB image to YCbCr
    """

    def __init__(self):
        super(RGB2YCbCrJpeg, self).__init__()
        matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
                          dtype=np.float32).T
        paddle.disable_static()
        self.matrix = paddle.Tensor(matrix)#paddle.create_parameter([3, 3], dtype='float32', default_initializer=matrix)
        # ww = np.array([0., 128., 128.], dtype=np.float32)
        self.shift = paddle.Tensor(np.array([0., 128., 128.], dtype=np.float32))#paddle.create_parameter([1, 3], dtype='float32', default_initializer=ww)

    def forward(self, image):
        """
        Args:
            image(Tensor): batch x 3 x height x width

        Returns:
            Tensor: batch x height x width x 3
        """
        image = image.transpose([0, 2, 3, 1])
        result = paddle.tensordot(image, self.matrix, axes=1)
        result = result + self.shift
        return result.reshape(image.shape)


class ChromaSubsampling(nn.Layer):
    """ Chroma subsampling on CbCr channels
    """

    def __init__(self):
        super(ChromaSubsampling, self).__init__()

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3

        Returns:
            y(tensor): batch x height x width
            cb(tensor): batch x height/2 x width/2
            cr(tensor): batch x height/2 x width/2
        """
        image_2 = image.transpose([0, 3, 1, 2]).clone()
        cb = F.avg_pool2d(image_2[:, 1, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2))
        cr = F.avg_pool2d(image_2[:, 2, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2))
        cb = cb.transpose([0, 2, 3, 1])
        cr = cr.transpose([0, 2, 3, 1])
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class BlockSplitting(nn.Layer):
    """ Splitting image into patches
    """

    def __init__(self):
        super(BlockSplitting, self).__init__()
        self.k = 8

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor:  batch x h*w/64 x h x w
        """
        height, _ = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.reshape([batch_size, height // self.k, self.k, -1, self.k])
        image_transposed = image_reshaped.transpose([0, 1, 3, 2, 4])
        return image_transposed.reshape([batch_size, -1, self.k, self.k])


class DCT8x8(nn.Layer):
    """ Discrete Cosine Transformation
    """

    def __init__(self):
        super(DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        tensor = paddle.Tensor(tensor)
        self.tensor = paddle.create_parameter(tensor.shape,
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(tensor))
        scale = paddle.Tensor(np.outer(alpha, alpha) * 0.25)
        self.scale = paddle.create_parameter(scale.shape, dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(scale))

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        image = image - 128
        result = self.scale * paddle.tensordot(image, self.tensor, axes=2)
        result.reshape(image.shape)
        return result


class YQuantize(nn.Layer):
    """ JPEG Quantization for Y channel

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = y_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            image = image.astype(paddle.float32) / (self.y_table * factor)
        else:
            b = factor.shape[0]
            table = self.y_table.expand([b, 1, 8, 8]) * factor.reshape([b, 1, 1, 1])
            image = image.astype(paddle.float32) / table
        image = self.rounding(image)
        return image


class CQuantize(nn.Layer):
    """ JPEG Quantization for CbCr channels

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = c_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            image = image.astype(paddle.float32) / (self.c_table * factor)
        else:
            b = factor.shape[0]
            table = self.c_table.expand([b, 1, 8, 8]) * factor.reshape([b, 1, 1, 1])
            image = image.astype(paddle.float32) / table
        image = self.rounding(image)
        return image


class CompressJpeg(nn.Layer):
    """Full JPEG compression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=paddle.round):
        super(CompressJpeg, self).__init__()
        self.l1 = nn.Sequential(RGB2YCbCrJpeg(), ChromaSubsampling())
        self.l2 = nn.Sequential(BlockSplitting(), DCT8x8())
        self.c_quantize = CQuantize(rounding=rounding)
        self.y_quantize = YQuantize(rounding=rounding)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x 3 x height x width

        Returns:
            dict(tensor): Compressed tensor with batch x h*w/64 x 8 x 8.
        """
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp, factor=factor)
            else:
                comp = self.y_quantize(comp, factor=factor)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


# ------------------------ decompression ------------------------#


class YDequantize(nn.Layer):
    """Dequantize Y channel
    """

    def __init__(self):
        super(YDequantize, self).__init__()
        self.y_table = y_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """

        if isinstance(factor, (int, float)):
            out = image * (self.y_table * factor)
        else:
            b = factor.shape[0]
            table = self.y_table.expand([b, 1, 8, 8]) * factor.reshape([b, 1, 1, 1])
            out = image * table
        return out


class CDequantize(nn.Layer):
    """Dequantize CbCr channel
    """

    def __init__(self):
        super(CDequantize, self).__init__()
        self.c_table = c_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            out = image * (self.c_table * factor)
        else:
            b = factor.shape[0]
            table = self.c_table.expand([b, 1, 8, 8]) * factor.reshape([b, 1, 1, 1])
            out = image * table
        return out


class iDCT8x8(nn.Layer):
    """Inverse discrete Cosine Transformation
    """

    def __init__(self):
        super(iDCT8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        alpha = paddle.Tensor(np.outer(alpha, alpha))
        self.alpha = paddle.create_parameter(alpha.shape, dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(alpha))
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        tensor = paddle.Tensor(tensor)
        self.tensor = paddle.create_parameter(tensor.shape, dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(tensor))

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """

        image = image * self.alpha
        result = 0.25 * paddle.tensordot(image, self.tensor, axes=2) + 128
        result.reshape(image.shape)
        return result


class BlockMerging(nn.Layer):
    """Merge patches into image
    """

    def __init__(self):
        super(BlockMerging, self).__init__()

    def forward(self, patches, height, width):
        """
        Args:
            patches(tensor) batch x height*width/64, height x width
            height(int)
            width(int)

        Returns:
            Tensor: batch x height x width
        """
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.reshape([batch_size, height // k, width // k, k, k])
        image_transposed = image_reshaped.transpose([0, 1, 3, 2, 4])
        return image_transposed.reshape([batch_size, height, width])


class ChromaUpsampling(nn.Layer):
    """Upsample chroma layers
    """

    def __init__(self):
        super(ChromaUpsampling, self).__init__()

    def forward(self, y, cb, cr):
        """
        Args:
            y(tensor): y channel image
            cb(tensor): cb channel
            cr(tensor): cr channel

        Returns:
            Tensor: batch x height x width x 3
        """

        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = paddle.tile(x,repeat_times=(1, 1, k, k))
            x = x.reshape([-1, height * k, width * k])
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        return paddle.concat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], axis=3)


class YCbCr2RGBJpeg(nn.Layer):
    """Converts YCbCr image to RGB JPEG
    """

    def __init__(self):
        super(YCbCr2RGBJpeg, self).__init__()

        matrix = np.array([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T
        shift = paddle.Tensor(np.array([0, -128., -128.]))
        self.shift = paddle.create_parameter(shift.shape, 'float32', default_initializer=paddle.nn.initializer.Assign(shift))
        matrix = paddle.Tensor(matrix)
        self.matrix = paddle.create_parameter(matrix.shape, 'float32', default_initializer=paddle.nn.initializer.Assign(matrix))

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3

        Returns:
            Tensor: batch x 3 x height x width
        """
        result = paddle.tensordot(image + self.shift, self.matrix, axes=1)
        return result.reshape(image.shape).transpose([0, 3, 1, 2])


class DeCompressJpeg(nn.Layer):
    """Full JPEG decompression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=paddle.round):
        super(DeCompressJpeg, self).__init__()
        self.c_dequantize = CDequantize()
        self.y_dequantize = YDequantize()
        self.idct = iDCT8x8()
        self.merging = BlockMerging()
        self.chroma = ChromaUpsampling()
        self.colors = YCbCr2RGBJpeg()

    def forward(self, y, cb, cr, imgh, imgw, factor=1):
        """
        Args:
            compressed(dict(tensor)): batch x h*w/64 x 8 x 8
            imgh(int)
            imgw(int)
            factor(float)

        Returns:
            Tensor: batch x 3 x height x width
        """
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                height, width = int(imgh / 2), int(imgw / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                height, width = imgh, imgw
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = paddle.clip(image, 0, 255)
        # image = paddle.min(255 * paddle.ones_like(image), paddle.max(paddle.zeros_like(image), image))
        return image / 255


# ------------------------ main DiffJPEG ------------------------ #


class DiffJPEG(nn.Layer):
    """This JPEG algorithm result is slightly different from cv2.
    DiffJPEG supports batch processing.

    Args:
        differentiable(bool): If True, uses custom differentiable rounding function, if False, uses standard torch.round
    """

    def __init__(self, differentiable=True):
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = paddle.round

        self.compress = CompressJpeg(rounding=rounding)
        self.decompress = DeCompressJpeg(rounding=rounding)

    def forward(self, x, quality):
        """
        Args:
            x (Tensor): Input image, bchw, rgb, [0, 1]
            quality(float): Quality factor for jpeg compression scheme.
        """
        factor = quality
        if isinstance(factor, (int, float)):
            factor = quality_to_factor(factor)
        else:
            for i in range(factor.shape[0]):
                factor[i] = quality_to_factor(factor[i])
        h, w = x.shape[-2:]
        h_pad, w_pad = 0, 0
        # why should use 16
        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)

        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress(y, cb, cr, (h + h_pad), (w + w_pad), factor=factor)
        recovered = recovered[:, :, 0:h, 0:w]
        return recovered


if __name__ == '__main__':
    import cv2

    from basicsr.utils import img2tensor, tensor2img

    img_gt = cv2.imread('test.png') / 255.

    # -------------- cv2 -------------- #
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
    _, encimg = cv2.imencode('.jpg', img_gt * 255., encode_param)
    img_lq = np.float32(cv2.imdecode(encimg, 1))
    cv2.imwrite('cv2_JPEG_20.png', img_lq)

    # -------------- DiffJPEG -------------- #
    jpeger = DiffJPEG(differentiable=False)
    img_gt = img2tensor(img_gt)
    img_gt = paddle.stack([img_gt, img_gt])
    quality = img_gt.new_tensor([20, 40])
    out = jpeger(img_gt, quality=quality)

    cv2.imwrite('pt_JPEG_20.png', tensor2img(out[0]))
    cv2.imwrite('pt_JPEG_40.png', tensor2img(out[1]))
