import paddle.nn as nn
import paddle.nn.functional as F 



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias_attr =False),
            nn.BatchNorm2D(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.LeakyReLU(0.1)
        )


def predict_flow(in_planes):
    return nn.Conv2D(in_planes,2,kernel_size=3,stride=1,padding=1,bias_attr =False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2DTranspose(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias_attr =False),
        nn.LeakyReLU(0.1)
    )


def crop_like(inpute, target):
    # print(inpute.shape)
    # print(target.shape)
    if inpute.shape[2:] == target.shape[2:]:
        return inpute
    else:
        return inpute[:, :, :target.shape[2], :target.shape[3]]
# def correlate(input1, input2):
#     out_corr = spatial_correlation_sample(input1,
#                                           input2,
#                                           kernel_size=1,
#                                           patch_size=21,
#                                           stride=1,
#                                           padding=0,
#                                           dilation_patch=2)
#     # collate dimensions 1 and 2 in order to be treated as a
#     # regular 4D tensor
#     b, ph, pw, h, w = out_corr.size()
#     out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
#     return F.leaky_relu_(out_corr, 0.1)
