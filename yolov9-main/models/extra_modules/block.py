# Powered bu https://blog.csdn.net/StopAndGoyyy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from models.common import *
from models.extra_modules.conv import *
from models.extra_modules.down import *
from models.extra_modules.attention import *
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


__all__ = ['AIFI', 'RepNCSPELAN4AKConv', 'Bifpn', 'C2f', 'C3', 'C2f_Attention', 'C3Ghost', 'C2fGhost', 'C2fSCConv',
           'C3SCConv', 'SimFusion_3in', 'SimFusion_4in', 'InjectionMultiSum_Auto_pool', 'PyramidPoolAgg',
           'TopBasicLayer', 'AdvPoolFusion', 'IFM', 'HGBlock', 'HWD_ADown', 'InceptionV1', 'InceptionV2a',
           'RepNCSPELAN4DySnakeConv', 'RepNCSPELAN4LSK', 'RepNCSPELAN4SCConv', 'SPD_ADown', 'C3Attention',
           'C3ScConv', 'C2fScConv', 'RepNCSPELAN4iRMB', 'Rep_SC_Atten', 'C2f_LSK', 'C3_LSK', 'RepNCSPELAN4_CD_LSK',
           'C3_CD_LSK', 'C2f_CD_LSK', 'InceptionV2b', 'InceptionV2c', 'InceptionV3a', 'InceptionV3b', 'InceptionV3c',
           'InceptionV3d', 'InceptionV3e', 'Stem', 'InceptionA', 'InceptionB', 'InceptionC', 'RedutionA', 'RedutionB',
           'GhostInceptionV1', 'GhostInceptionV2c', 'GhostInceptionV2a', 'GhostInceptionV2b'

           ]


# -------------------------C2fScConv----------------------------------
class C2f_CD_LSK(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.cd = CD_LSK(self.c, 2 * self.c)

    def forward(self, x):
        """Forward pass through C2f layer."""
        x1, x2 = self.cv1(x).chunk(2, 1)
        y = [self.cd(x1), x2]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# -------------------------C3_CD_LSK----------------------------------
class C3_CD_LSK(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cd = CD_LSK(c_, 2 * c_)

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cd(self.cv2(x))), 1))


# -------------------------RepNCSPELAN4_CD_LSK----------------------------------
class RepNCSPELAN4_CD_LSK(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)
        self.cd = CD_LSK(self.c, 2*self.c)

    def forward(self, x):
        x1, x2 = list(self.cv1(x).chunk(2, 1))
        y = [self.cd(x1), x2]
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# -------------------------C3ScConv----------------------------------
class C3_LSK(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BottleneckWithLSKAttention(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


# -------------------------C2f_LSK----------------------------------
class BottleneckWithLSKAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = LSKblock(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C2f_LSK(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            BottleneckWithLSKAttention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


# -------------------------RepNCSPELAN4iRMB----------------------------------
class RepNCSPELAN4iRMB(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, out number of c2, out number of c3, RepNCSP number
        super().__init__(c1, c2, c3, c4, c5)
        self.cv5 = iRMB(c2, c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv5(self.cv4(torch.cat(y, 1)))


# -------------------------C3ScConv----------------------------------
class Bottleneck_ScConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c2)


class C3ScConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


# -------------------------C2fScConv----------------------------------

class C2fScConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ScConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


# -------------------------C3Attention----------------------------------
class C3Attention(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BottleneckByAttention(c_, c_) for _ in range(n)))


# -------------------------SPD_ADown____________________
class SPD_ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = SPDConv(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 3, 1, 1, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


# -------------------------RepNCSPELAN4SCConv____________________
class RepNBottleneck_SC(RepNBottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__( c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = SCConv(c_, c2, s=1, g=g)
        self.add = shortcut and c1 == c2


class RepNCSP_SCConv(RepNCSP):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_)
        self.cv2 = SCConv(c1, c_)
        self.cv3 = Conv(2 * c_, c2)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck_SC(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4SCConv(RepNCSPELAN4):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, c3, c4, c5)
        self.cv1 = Conv(c1, c3, k=1, s=1)
        self.cv2 = nn.Sequential(RepNCSP_SCConv(c3 // 2, c4, c5), SCConv(c4, c4))
        self.cv3 = nn.Sequential(RepNCSP_SCConv(c4, c4, c5), SCConv(c4, c4))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class Rep_SC_Atten(RepNCSPELAN4SCConv):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, c3, c4, c5)
        self.cv4 = Conv(c3 + (2 * c4), 2*c2, 1, 1)
        self.cv5 = Conv(2*c2, c2//2, k=1, s=1)
        self.cv6 = iRMB(c2//2, c2//2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        y = self.cv5(self.cv4(torch.cat(y, 1)))
        return torch.cat((y, self.cv6(y)), dim=1)


# -------------------------RepNCSPELAN4LSKDeep----------------------------------
class RepNCSP_LSK(RepNCSP):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_)
        self.cv2 = Conv(c1, c_)
        self.cv3 = Conv(2 * c_, c2)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.atten = LSKblock(c2)

    def forward(self, x):
        return self.atten(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))


class RepNCSPELAN4LSK(RepNCSPELAN4):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, c3, c4, c5)
        self.cv1 = Conv(c1, c3, k=1, s=1)
        self.cv2 = nn.Sequential(RepNCSP_LSK(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP_LSK(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)


# -------------------------RepNCSPELAN4DySnakeConv____________________
class RepNBottleneck_DySnakeConv(RepNBottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], s=1, g=g)
        self.add = shortcut and c1 == c2


class RepNCSP_DySnakeConv(RepNCSP):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DySnakeConv(c1, c_)
        self.cv2 = DySnakeConv(c1, c_)
        self.cv3 = DySnakeConv(2 * c_, c2)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4DySnakeConv(RepNCSPELAN4):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, c3, c4, c5)
        self.cv1 = Conv(c1, c3, k=1, s=1)
        self.cv2 = nn.Sequential(RepNCSP_DySnakeConv(c3 // 2, c4, c5), DySnakeConv(c4, c4, 3))
        self.cv3 = nn.Sequential(RepNCSP_DySnakeConv(c4, c4, c5), DySnakeConv(c4, c4, 3))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


# -------------------------InceptionV1----------------------------------
class InceptionV1(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_out, 1, s=1, act=act), Conv(c_out, c_out, 5, s=s, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class GhostInceptionV1(InceptionV1):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__(c1, c2, s, act)
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = GhostConv(c1, c_, 1, s=s, act=act)
        self.conv2 = nn.Sequential(GhostConv(c1, c_, 1, s=1, act=act), GhostConv(c_, c_, 3, s=s, act=act))
        self.conv3 = nn.Sequential(GhostConv(c1, c_out, 1, s=1, act=act), GhostConv(c_out, c_out, 5, s=s, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), GhostConv(c1, c_, 1, s=1, act=act))


# -------------------------InceptionV2----------------------------------
class InceptionV2a(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_out, c_out, 3, s=s, act=act),
                                   Conv(c_out, c_out, 3, s=1, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class GhostInceptionV2a(InceptionV2a):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__(c1, c2, s, act)
        c_ = c2 // 8 * 2
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(GhostConv(c1, c_, 1, s=1, act=act), GhostConv(c_, c_, 3, s=s, act=act))
        self.conv3 = nn.Sequential(GhostConv(c1, c_, 1, s=1, act=act), GhostConv(c_out, c_out, 3, s=s, act=act),
                                   GhostConv(c_out, c_out, 3, s=1, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), GhostConv(c1, c_, 1, s=1, act=act))


class InceptionV2b(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (3, 1), s=1, act=act),
                                   Conv(c_, c_, (1, 3), s=1, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (3, 1), s=1, act=act),
                                   Conv(c_, c_, (1, 3), s=1, act=act), Conv(c_, c_, (3, 1), s=1, act=act),
                                   Conv(c_, c_, (1, 3), s=1, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class GhostInceptionV2b(InceptionV2b):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__(c1, c2, s, act)
        c_ = c2 // 8 * 2
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(GhostConv(c1, c_, 1, s=s, act=act), GhostConv(c_, c_, (3, 1), s=1, act=act),
                                   GhostConv(c_, c_, (1, 3), s=1, act=act))
        self.conv3 = nn.Sequential(GhostConv(c1, c_, 1, s=s, act=act), GhostConv(c_, c_, (3, 1), s=1, act=act),
                                   GhostConv(c_, c_, (1, 3), s=1, act=act), GhostConv(c_, c_, (3, 1), s=1, act=act),
                                   GhostConv(c_, c_, (1, 3), s=1, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), GhostConv(c1, c_, 1, s=1, act=act))


class InceptionV2c(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 6
        c_out = c2 - 5 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = Conv(c1, c_, 1, s=s, act=act)
        self.conv2_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv2_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, act=act))
        self.conv3_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv3_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return torch.cat([self.conv1(x), self.conv2_1(x2), self.conv2_2(x2), self.conv3_1(x3), self.conv3_2(x3), self.pool(x)], dim=1)


class GhostInceptionV2c(InceptionV2c):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__(c1, c2, s, act)
        c_ = c2 // 12 * 2
        c_out = c2 - 5 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = GhostConv(c1, c_, 1, s=s, act=act)
        self.conv2_1 = GhostConv(c_, c_, (3, 1), s=1, act=act)
        self.conv2_2 = GhostConv(c_, c_, (1, 3), s=1, act=act)
        self.conv3 = nn.Sequential(GhostConv(c1, c_, 1, s=1, act=act), GhostConv(c_, c_, 3, s=s, act=act))
        self.conv3_1 = GhostConv(c_, c_, (3, 1), s=1, act=act)
        self.conv3_2 = GhostConv(c_, c_, (1, 3), s=1, act=act)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), GhostConv(c1, c_, 1, s=1, act=act))


# -------------------------InceptionV3----------------------------------
# https://zhuanlan.zhihu.com/p/30172532
class InceptionV3a(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 5, s=s, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_out, c_out, 3, s=s, act=act),
                                   Conv(c_out, c_out, 3, s=1, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class InceptionV3b(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 3
        c_out = c2 - 2*c_
        self.conv1 = Conv(c1, c_out, 3, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, 3, s=1, act=act),
                                   Conv(c_, c_, 3, s=1, act=act))
        self.pool = Conv(c1, c_, 5, s=s, act=act)

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.pool(x)], dim=1)


class InceptionV3c(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=s, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=s, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=s, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class InceptionV3d(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 3
        c_out = c2 - 2*c_
        self.conv1 = nn.Sequential(Conv(c1, c_out, 1, s=s, act=act), Conv(c_out, c_out, 3, s=1, act=act))
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=1, act=act),
                                   Conv(c_, c_, 3, s=1, act=act), Conv(c_, c_, 3, s=s, act=act))
        self.pool = Conv(c1, c_, 5, s=s, act=act)

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.pool(x)], dim=1)


class InceptionV3e(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 6
        c_out = c2 - 5 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = Conv(c1, c_, 1, s=s, act=act)
        self.conv2_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv2_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, act=act))
        self.conv3_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv3_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return torch.cat([self.conv1(x), self.conv2_1(x2), self.conv2_2(x2), self.conv3_1(x3), self.conv3_2(x3), self.pool(x)], dim=1)


# -------------------------InceptionV4----------------------------------
# https://blog.csdn.net/Next_SummerAgain/article/details/129835944?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-129835944-blog-107676194.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.1&utm_relevant_index=3
class Stem(nn.Module):
    def __init__(self, c1, scale=1):
        super().__init__()
        ch1 = [32, 32, 64, 96,] * scale
        ch2_1 = [64, 96,] * scale
        ch2_2 = [64, 64, 64, 96,] * scale
        ch3 = [192,] * scale
        self.conv1 = nn.Sequential(Conv(c1, ch1[0], 3, s=2), Conv(ch1[0], ch1[1], 3), Conv(ch1[1], ch1[2], 3, s=1),
                                   )
        self.conv1_1 = nn.MaxPool2d(3, 2, 1)
        self.conv1_2 = Conv(ch1[2], ch1[3], 3, s=2)
        self.conv2_1 = nn.Sequential(Conv(ch1[3] + ch1[2], ch2_1[0], 1), Conv(ch2_1[0], ch2_1[1], 3))
        self.conv2_2 = nn.Sequential(Conv(ch1[3] + ch1[2], ch2_2[0], 1), Conv(ch2_2[0], ch2_2[1], (7, 1)),
                                     Conv(ch2_2[1], ch2_2[2], (1, 7)), Conv(ch2_2[2], ch2_2[3], 3))
        self.conv3_1 = Conv(ch2_2[3] + ch2_1[1], ch3[0], 3, 2)
        self.conv3_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = torch.cat([self.conv1_1(self.conv1(x)), self.conv1_2(self.conv1(x))], dim=1)
        x2 = torch.cat([self.conv2_1(x1), self.conv2_2(x1)], dim=1)
        x3 = torch.cat([self.conv3_1(x2), self.conv3_2(x2)], dim=1)
        return x3


class InceptionA(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=s, act=act),
                                   Conv(c_, c_, 3, s=1, act=act))
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, act=act))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class InceptionB(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=s, act=act))
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=s, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=s, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)], dim=1)


class InceptionC(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        c_ = c2 // 6
        c_out = c2 - 5 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = Conv(c1, c_, 1, s=s, act=act)
        self.conv2_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv2_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.conv3 = nn.Sequential(Conv(c1, c_, 1, s=s, act=act), Conv(c_, c_, (1, 3), s=1, act=act),
                                   Conv(c_, c_, (3, 1), s=1, act=act))
        self.conv3_1 = Conv(c_, c_, (3, 1), s=1, act=act)
        self.conv3_2 = Conv(c_, c_, (1, 3), s=1, act=act)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), Conv(c1, c_, 1, s=1, act=act))

    def forward(self, x):
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return torch.cat([self.conv1(x), self.conv2_1(x2), self.conv2_2(x2), self.conv3_1(x3), self.conv3_2(x3), self.pool(x)], dim=1)


class RedutionA(nn.Module):
    def __init__(self, c1, c2, s=2, act=True):
        super().__init__()
        c_ = c2 // 3
        c_out = c2 - 2*c_
        self.conv1 = Conv(c1, c_out, 3, s=s, act=act)
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, 3, s=1, act=act),
                                   Conv(c_, c_, 3, s=s, act=act))
        self.pool = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), nn.MaxPool2d(3, s, 1))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.pool(x)], dim=1)


class RedutionB(nn.Module):
    def __init__(self, c1, c2, s=2, act=True):
        super().__init__()
        c_ = c2 // 3
        c_out = c2 - 2*c_
        self.conv1 = nn.Sequential(Conv(c1, c_out, 1, s=1, act=act), Conv(c_out, c_out, 3, s=s, act=act))
        self.conv2 = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), Conv(c_, c_, (1, 7), s=1, act=act),
                                   Conv(c_, c_, (7, 1), s=1, act=act), Conv(c_, c_, k=3, s=s, act=act))
        self.pool = nn.Sequential(Conv(c1, c_, 1, s=1, act=act), nn.MaxPool2d(3, s, 1))

    def forward(self, x):

        return torch.cat([self.conv1(x), self.conv2(x), self.pool(x)], dim=1)


# -------------------------C2f_Attention----------------------------------
class BottleneckByAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = SE(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C2f_Attention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            BottleneckByAttention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


# -------------------------HWD_ADown____________________
class HWD_ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = HWD(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


# -------------------------HGBlock----------------------------------
class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=True):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


# -------------------------Gold YOLO----------------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                2] != out_channels else nn.Identity()
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)

        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        x0 = self.cv1(self.downsample(x[0], output_size))
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])

        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out


class IFM(nn.Module):
    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc))
        )

    def forward(self, x):
        return self.conv(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp: list,
            flag: int
    ) -> None:
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(gloabl_info)
        global_feat = self.global_embedding(gloabl_info)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class PyramidPoolAgg(nn.Module):
    def __init__(self, inc, ouc, stride, pool_mode='torch'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
        self.conv = Conv(inc, ouc)

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return self.conv(torch.cat(out, dim=1))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = Conv(hidden_features, out_features, act=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GOLDYOLO_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv(dim, nh_kd, 1, act=False)
        self.to_k = Conv(dim, nh_kd, 1, act=False)
        self.to_v = Conv(dim, self.dh, 1, act=False)

        self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = GOLDYOLO_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(self, embedding_dim, ouc_list, block_num=2, key_dim=8, num_heads=4,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path))
        self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return self.conv(x)


class AdvPoolFusion(nn.Module):
    def forward(self, x):
        x1, x2 = x
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)


# -------------------------C3Ghost----------------------------------
class C3SCConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SCConv(c_, c_) for _ in range(n)))


# -------------------------C2fSCConv----------------------------------
class Bottleneck_SCConv(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SCConv(c1, c_)
        self.cv2 = SCConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2fSCConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_SCConv(self.c , self.c , shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# -------------------------C2fGhost____________________
class C2fGhost(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(GhostBottleneckC2f(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


class GhostBottleneckC2f(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, k[0], 1)
        self.cv2 = GhostConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

# -------------------------C3Ghost____________________
class C3Ghost(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1, act=False),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# -------------------------C2f_Attention----------------------------------
class BottleneckByAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = SE(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C2f_Attention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            BottleneckByAttention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


# -------------------------C3----------------------------------
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# -------------------------C2f----------------------------------
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# -------------------------Bifpn----------------------------------
class Bifpn(nn.Module):
    def __init__(self, inc_list):
        super().__init__()
        self.layer_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        layer_weight = self.relu(self.layer_weight.clone())
        layer_weight = layer_weight / (torch.sum(layer_weight, dim=0))
        return torch.sum(torch.stack([layer_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)


# -------------------------RepNCSPELAN4AKConv____________________
class RepNBottleneck_AKConv(RepNBottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = AKConv(c_, c2, 1)
        self.add = shortcut and c1 == c2


class RepNCSP_AKConv(RepNCSP):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck_AKConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4AKConv(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, out number of c2, out number of c3, RepNCSP number
        super().__init__(c1, c2, c3, c4, c5)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP_AKConv(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP_AKConv(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


# -------------------------AIFI----------------------------------
class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()

        TORCH_1_9 = check_version(torch.__version__, '1.9.0')

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


if __name__ == '__main__':
    module_test = GhostInceptionV2c(32, 64, 2)
    print(module_test(torch.zeros(3, 32, 640, 480)).shape)
