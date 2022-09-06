
from paddle import nn as nn
from paddle.nn import Conv2D
from paddle.nn import functional as F
from paddle.nn import SpectralNorm
from paddle.nn.utils import spectral_norm

class UNetDiscriminatorSN2(nn.Layer):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN2, self).__init__()
        self.skip_connection = skip_connection

        self.conv0 = Conv2D(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = Conv2D(num_feat, num_feat * 2, 4, 2, 1, bias_attr=False)
        self.conv2 = Conv2D(num_feat * 2, num_feat * 4, 4, 2, 1, bias_attr=False)
        self.conv3 = Conv2D(num_feat * 4, num_feat * 8, 4, 2, 1, bias_attr=False)
        # upsample
        self.conv4 = Conv2D(num_feat * 8, num_feat * 4, 3, 1, 1, bias_attr=False)
        self.conv5 = Conv2D(num_feat * 4, num_feat * 2, 3, 1, 1, bias_attr=False)
        self.conv6 = Conv2D(num_feat * 2, num_feat, 3, 1, 1, bias_attr=False)

        # extra
        self.conv7 = Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=False)
        self.conv8 = Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=False)

        self.conv9 = Conv2D(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        x1 = self.conv1(x0)
        norm = SpectralNorm(x1.shape, dim=1, power_iters=2)
        x1 = norm(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.conv2(x1)
        norm = SpectralNorm(x2.shape, dim=1, power_iters=2)
        x2 = norm(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.2)
        x3 = self.conv3(x2)
        norm = SpectralNorm(x3.shape, dim=1, power_iters=2)
        x3 = norm(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.2)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = self.conv4(x3)
        norm = SpectralNorm(x4.shape, dim=1, power_iters=2)
        x4 = norm(x4)
        x4 = F.leaky_relu(x4, negative_slope=0.2)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = self.conv5(x4)
        norm = SpectralNorm(x5.shape, dim=1, power_iters=2)
        x5 = norm(x5)
        x5 = F.leaky_relu(x5, negative_slope=0.2)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = self.conv6(x5)
        norm = SpectralNorm(x6.shape, dim=1, power_iters=2)
        x6 = norm(x6)
        x6 = F.leaky_relu(x6, negative_slope=0.2)

        if self.skip_connection:
            x6 = x6 + x0

        # extra
        out = self.conv7(x6)
        norm = SpectralNorm(out.shape, dim=1, power_iters=2)
        out = norm(out)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.conv8(out)
        norm = SpectralNorm(out.shape, dim=1, power_iters=2)
        out = norm(out)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.conv9(out)

        return out




class UNetDiscriminatorSN(nn.Layer):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = norm(Conv2D(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1))
        # downsample
        self.conv1 = norm(Conv2D(num_feat, num_feat * 2, 4, 2, 1, bias_attr=False))
        self.conv2 = norm(Conv2D(num_feat * 2, num_feat * 4, 4, 2, 1, bias_attr=False))
        self.conv3 = norm(Conv2D(num_feat * 4, num_feat * 8, 4, 2, 1, bias_attr=False))
        # upsample
        self.conv4 = norm(Conv2D(num_feat * 8, num_feat * 4, 3, 1, 1, bias_attr=False))
        self.conv5 = norm(Conv2D(num_feat * 4, num_feat * 2, 3, 1, 1, bias_attr=False))
        self.conv6 = norm(Conv2D(num_feat * 2, num_feat, 3, 1, 1, bias_attr=False))
        # extra
        self.conv7 = norm(Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=False))
        self.conv8 = norm(Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=False))
        self.conv9 = Conv2D(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2)
        out = self.conv9(out)

        return out
