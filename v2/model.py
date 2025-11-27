import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

# ==================== Attention Modules ====================

class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA) - Lightweight attention"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Adaptive kernel size calculation
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x)
        # Exchange dimensions for 1D conv: [B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D convolution across channels
        y = self.conv(y)
        # Exchange back: [B, 1, C] -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Excitation with sigmoid
        y = self.sigmoid(y)
        # Scale original features
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (implemented as Conv2d)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))
        # Max pooling path
        max_out = self.fc(self.max_pool(x))
        # Combine both paths
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        y = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid
        y = self.sigmoid(self.conv(y))
        return y


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_att(x)
        # Apply spatial attention
        x = x * self.spatial_att(x)
        return x


# ==================== Basic Building Blocks ====================

class PreLayer3x3(nn.Module):
    """3x3 SRM preprocessing layer"""
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
    """5x5 SRM preprocessing layer"""
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


class PreprocessingLayer(nn.Module):
    """Preprocessing layer combining 3x3 and 5x5 SRM filters"""
    def __init__(self):
        super(PreprocessingLayer, self).__init__()
        self.conv1 = PreLayer3x3()
        self.conv2 = PreLayer5x5()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)


class SPPLayer(nn.Module):
    """Spatial Pyramid Pooling layer"""
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
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
    """Basic convolutional block with BatchNorm and ReLU"""
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


# ==================== Main Model with CBAM and ECA ====================

class ConvLayerWithAttention(nn.Module):
    """
    ConvLayer enhanced with both CBAM and ECA blocks.

    Strategy:
    - ECA for early stages (lightweight, efficient)
    - CBAM for later stages (powerful, captures spatial patterns)
    """
    def __init__(self, reduction=16):
        super(ConvLayerWithAttention, self).__init__()

        # Depthwise separable convolutions (original residual blocks)
        self.conv1 = nn.Conv2d(30, 60, 3, 1, 1, groups=30)
        self.conv1_1 = nn.Conv2d(60, 30, 1)
        self.bn1 = nn.BatchNorm2d(30)

        self.conv2 = nn.Conv2d(30, 60, 3, 1, 1, groups=30)
        self.conv2_1 = nn.Conv2d(60, 30, 1)
        self.bn2 = nn.BatchNorm2d(30)

        # ECA after residual blocks (lightweight)
        self.eca_residual = ECABlock(30)

        # Stage 1: Basic conv + ECA (early stage, keep it light)
        self.stage1 = BasicBlock(30, 32, 3, 1, 1, False)
        self.eca1 = ECABlock(32)

        # Stage 2: Basic conv + ECA
        self.stage2 = BasicBlock(32, 32, 3, 1, 1, False)
        self.eca2 = ECABlock(32)

        # Stage 3: Basic conv + CBAM (deeper stage, more powerful attention)
        self.stage3 = BasicBlock(32, 64, 3, 1, 1, False)
        self.cbam3 = CBAMBlock(64, reduction=reduction)

        # Stage 4: Basic conv + CBAM (deepest stage, full attention)
        self.stage4 = BasicBlock(64, 128, 3, 1, 1, True)
        self.cbam4 = CBAMBlock(128, reduction=reduction)

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
        """Initialize convolutional and batch norm layers"""
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
        x = self.conv1(x).abs()  # Absolute value for steganalysis
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Second residual block with skip connection
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.bn2(x)
        x += temp  # Residual connection
        x = F.relu(x, inplace=True)

        # ECA attention after residual blocks
        x = self.eca_residual(x)

        # Stage 1: Conv + ECA
        x = self.stage1(x)
        x = self.eca1(x)

        # Stage 2: Conv + ECA
        x = self.stage2(x)
        x = self.eca2(x)

        # Stage 3: Conv + CBAM (spatial patterns become important)
        x = self.stage3(x)
        x = self.cbam3(x)

        # Stage 4: Conv + CBAM (high-level features)
        x = self.stage4(x)
        x = self.cbam4(x)

        # Spatial pyramid pooling
        x = self.spp(x)

        # Classification
        x = self.classifier(x)

        return x


class AttentionZhuNet(nn.Module):
    """
    ZhuNet enhanced with CBAM and ECA attention mechanisms.

    Architecture:
    1. SRM preprocessing layer (3x3 and 5x5 filters)
    2. Convolutional layers with residual connections
    3. ECA attention on early stages (lightweight)
    4. CBAM attention on later stages (powerful)
    5. Spatial pyramid pooling
    6. Fully connected classifier
    """
    def __init__(self, reduction=16):
        super(AttentionZhuNet, self).__init__()
        self.layer1 = PreprocessingLayer()
        self.layer2 = ConvLayerWithAttention(reduction=reduction)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
