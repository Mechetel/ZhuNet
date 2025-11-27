import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np


class PreLayer3x3(nn.Module):
    """3x3 SRM preprocessing layer."""

    def __init__(self, stride=1, padding=1):
        super(PreLayer3x3, self).__init__()
        self.in_channels = 1
        self.out_channels = 25
        self.kernel_size = (3, 3)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(25, 1, 3, 3), requires_grad=True)
        self.bias = Parameter(torch.Tensor(25), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        # Load SRM kernels from numpy file
        srm_npy = np.load('kernels/SRM3_3.npy')
        self.weight.data.numpy()[:] = srm_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class PreLayer5x5(nn.Module):
    """5x5 SRM preprocessing layer."""

    def __init__(self, stride=1, padding=2):
        super(PreLayer5x5, self).__init__()
        self.in_channels = 1
        self.out_channels = 5
        self.kernel_size = (5, 5)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(5, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(5), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        # Load SRM kernels from numpy file
        srm_npy = np.load('kernels/SRM5_5.npy')
        self.weight.data.numpy()[:] = srm_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class SPPLayer(nn.Module):
    """Spatial Pyramid Pooling layer."""

    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        """
        Apply spatial pyramid pooling at multiple scales.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Flattened pooled features
        """
        batch_size, channels, height, width = x.size()

        # Level 1: Global average pooling (1x1)
        pool1 = F.avg_pool2d(x, (height, width), stride=(height, width))
        pool1 = pool1.view(batch_size, -1)

        # Level 2: 2x2 pooling
        pool2 = F.avg_pool2d(x, kernel_size=height // 2, stride=height // 2)
        pool2 = pool2.view(batch_size, -1)

        # Level 3: 4x4 pooling
        pool3 = F.avg_pool2d(x, kernel_size=height // 4, stride=height // 4)
        pool3 = pool3.view(batch_size, -1)

        # Concatenate all levels
        spp = torch.cat([pool1, pool2, pool3], dim=1)

        return spp


class BasicBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_global):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.use_global = use_global
        self.pool = nn.AvgPool2d(5, 2, 2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))

        if not self.use_global:
            x = self.pool(x)

        return x


class PreprocessingLayer(nn.Module):
    """Preprocessing layer combining 3x3 and 5x5 SRM filters."""

    def __init__(self):
        super(PreprocessingLayer, self).__init__()
        self.conv1 = PreLayer3x3()
        self.conv2 = PreLayer5x5()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)


class ConvLayer(nn.Module):
    """Main convolutional layers with residual connection and SPP."""

    def __init__(self):
        super(ConvLayer, self).__init__()

        # Depthwise separable convolutions
        self.conv1 = nn.Conv2d(30, 60, 3, 1, 1, groups=30)
        self.conv1_1 = nn.Conv2d(60, 30, 1)
        self.bn1 = nn.BatchNorm2d(30)

        self.conv2 = nn.Conv2d(30, 60, 3, 1, 1, groups=30)
        self.conv2_1 = nn.Conv2d(60, 30, 1)
        self.bn2 = nn.BatchNorm2d(30)

        # Main convolutional layers
        self.conv_layer = nn.Sequential(
            BasicBlock(30, 32, 3, 1, 1, False),
            BasicBlock(32, 32, 3, 1, 1, False),
            BasicBlock(32, 64, 3, 1, 1, False),
            BasicBlock(64, 128, 3, 1, 1, True)
        )

        # Spatial pyramid pooling
        self.spp = SPPLayer()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2688, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolutional and batch norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # First residual block
        temp = x
        x = self.conv1(x).abs()  # Apply absolute value to enhance residuals
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Second residual block with skip connection
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.bn2(x)
        x += temp  # Residual connection
        x = F.relu(x, inplace=True)

        # Main convolutional layers
        x = self.conv_layer(x)

        # Spatial pyramid pooling
        x = self.spp(x)

        # Classification
        x = self.classifier(x)

        return x


class ZhuNet(nn.Module):
    """
    Zhu-Net for image steganalysis.

    Architecture:
    1. SRM preprocessing layer (3x3 and 5x5 filters)
    2. Convolutional layers with residual connections
    3. Spatial pyramid pooling
    4. Fully connected classifier
    """

    def __init__(self):
        super(ZhuNet, self).__init__()
        self.layer1 = PreprocessingLayer()
        self.layer2 = ConvLayer()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
