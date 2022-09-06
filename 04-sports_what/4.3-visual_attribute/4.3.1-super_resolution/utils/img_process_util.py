import cv2
import numpy as np
import paddle
from paddle.nn import functional as F


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.shape[-1]
    b, c, h, w = img.shape
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.shape[-2:]

    if kernel.shape[0] == 1:
        # apply the same kernel to all batch images
        img = img.reshape([b * c, 1, ph, pw])
        kernel = kernel.reshape([1, 1, k, k])
        return F.conv2d(img.astype(paddle.float32), kernel.astype(paddle.float32), padding=0).reshape([b, c, h, w])
    else:
        img = img.reshape([1, b * c, ph, pw])
        kernel = paddle.tile(kernel.reshape([b, 1, k, k]),repeat_times=(1, c, 1, 1)).reshape([b * c, 1, k, k])
        return F.conv2d(img.astype(paddle.float32), kernel.astype(paddle.float32), groups=b * c).reshape([b, c, h, w])


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(paddle.nn.Layer):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = paddle.Tensor(np.dot(kernel, kernel.transpose()).astype(np.float32)).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = paddle.abs(residual) * 255 > threshold
        mask = mask.astype(paddle.float32)
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = paddle.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img

class USMSharp_npy():

    def __init__(self, radius=50, sigma=0):
        super(USMSharp_npy, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        self.kernel = np.dot(kernel, kernel.transpose()).astype(np.float32)

    def filt(self, img, weight=0.5, threshold=10):
        blur = cv2.filter2D(img, -1, self.kernel)
        residual = img - blur

        mask = np.abs(residual) * 255 > threshold
        mask = mask.astype(np.float32)
        soft_mask = cv2.filter2D(mask, -1, self.kernel)
        sharp = img + weight * residual
        sharp = np.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img

