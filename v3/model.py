import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np


# ==================== Advanced Attention Modules ====================

class ECABlock(nn.Module):
    """Efficient Channel Attention - Lightweight and effective"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Enhanced Channel Attention with both avg and max pooling"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention focusing on 'where' is important"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - Proven effective for steganalysis"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==================== SRM Preprocessing ====================

class PreLayer3x3(nn.Module):
    """3x3 SRM preprocessing layer"""
    def __init__(self, stride=1, padding=1):
        super(PreLayer3x3, self).__init__()
        self.weight = Parameter(torch.Tensor(25, 1, 3, 3), requires_grad=True)
        self.bias = Parameter(torch.Tensor(25), requires_grad=True)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.reset_parameters()

    def reset_parameters(self):
        srm_npy = np.load('kernels/SRM3_3.npy')
        self.weight.data.numpy()[:] = srm_npy
        self.bias.data.zero_()

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class PreLayer5x5(nn.Module):
    """5x5 SRM preprocessing layer"""
    def __init__(self, stride=1, padding=2):
        super(PreLayer5x5, self).__init__()
        self.weight = Parameter(torch.Tensor(5, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(5), requires_grad=True)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.reset_parameters()

    def reset_parameters(self):
        srm_npy = np.load('kernels/SRM5_5.npy')
        self.weight.data.numpy()[:] = srm_npy
        self.bias.data.zero_()

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class EnhancedSRMLayer(nn.Module):
    """Enhanced SRM with additional Bayar constraint layer"""
    def __init__(self):
        super(EnhancedSRMLayer, self).__init__()
        self.conv3x3 = PreLayer3x3()
        self.conv5x5 = PreLayer5x5()

        # Bayar's constrained convolutional layer (learned preprocessing)
        self.bayar_conv = nn.Conv2d(1, 3, kernel_size=5, padding=2, bias=False)
        self._init_bayar_weights()

    def _init_bayar_weights(self):
        """Initialize Bayar constraint: center must be -1, sum of others = 1"""
        with torch.no_grad():
            self.bayar_conv.weight.normal_(0, 0.01)
            # Set center to -1
            self.bayar_conv.weight[:, :, 2, 2] = -1.0

    def forward(self, x):
        # Apply constraint to Bayar layer
        with torch.no_grad():
            self.bayar_conv.weight[:, :, 2, 2] = -1.0
            # Normalize so sum of non-center weights = 1
            mask = torch.ones_like(self.bayar_conv.weight)
            mask[:, :, 2, 2] = 0
            self.bayar_conv.weight *= mask
            sum_weights = self.bayar_conv.weight.sum(dim=(2, 3), keepdim=True)
            self.bayar_conv.weight /= (sum_weights + 1e-8)
            self.bayar_conv.weight[:, :, 2, 2] = -1.0

        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.bayar_conv(x)
        return torch.cat([x1, x2, x3], dim=1)  # 30 + 3 = 33 channels


# ==================== Residual Blocks with Attention ====================

class ResidualBlock(nn.Module):
    """Enhanced Residual Block with SE or CBAM attention"""
    def __init__(self, in_channels, out_channels, stride=1, attention='se'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention mechanism
        if attention == 'se':
            self.attention = SEBlock(out_channels, reduction=16)
        elif attention == 'cbam':
            self.attention = CBAM(out_channels, reduction=16)
        elif attention == 'eca':
            self.attention = ECABlock(out_channels)
        else:
            self.attention = nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.attention(out)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class SRNetInspiredBlock(nn.Module):
    """SRNet-inspired convolutional block with modern improvements"""
    def __init__(self, in_channels, out_channels, use_pool=True):
        super(SRNetInspiredBlock, self).__init__()

        # Type 1: Regular convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Type 2: Separable convolution
        self.conv2_dw = nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                                   groups=out_channels, bias=False)
        self.conv2_pw = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention
        self.se = SEBlock(out_channels, reduction=16)

        # Pooling
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.AvgPool2d(3, 2, 1)

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.residual(x)

        # Path 1
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # Path 2
        out = self.conv2_dw(out)
        out = self.conv2_pw(out)
        out = self.bn2(out)

        # Attention
        out = self.se(out)

        # Add residual
        out += identity
        out = F.relu(out, inplace=True)

        # Pooling
        if self.use_pool:
            out = self.pool(out)

        return out


# ==================== Multi-Scale Feature Extraction ====================

class MultiScaleConv(nn.Module):
    """Multi-scale convolution for capturing features at different scales"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()

        # Different kernel sizes for multi-scale
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x1, x3, x5, x7], dim=1)
        return F.relu(self.bn(out), inplace=True)


# ==================== Main Model ====================

class AttentionZhuNet(nn.Module):
    """
    Improved Steganalysis Network with multiple enhancements:
    1. Enhanced SRM preprocessing with Bayar constraint layer
    2. SRNet-inspired residual blocks with SE attention
    3. Multi-scale feature extraction
    4. Progressive feature refinement
    5. Advanced attention mechanisms (SE, CBAM, ECA)
    6. Better feature aggregation
    """

    def __init__(self, num_classes=2):
        super(AttentionZhuNet, self).__init__()

        # Enhanced SRM preprocessing (33 channels)
        self.srm_layer = EnhancedSRMLayer()
        for param in self.srm_layer.parameters():
            param.requires_grad = False

        # Initial feature extraction with TLU (Truncated Linear Unit)
        self.init_conv = nn.Conv2d(33, 64, 3, 1, 1, bias=False)
        self.init_bn = nn.BatchNorm2d(64)

        # Stage 1: Light attention (64 -> 64)
        self.stage1 = nn.Sequential(
            SRNetInspiredBlock(64, 64, use_pool=True),
            ECABlock(64)
        )

        # Stage 2: Multi-scale features (64 -> 128)
        self.stage2 = nn.Sequential(
            MultiScaleConv(64, 128),
            SRNetInspiredBlock(128, 128, use_pool=True),
            SEBlock(128)
        )

        # Stage 3: Deep features (128 -> 256)
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, attention='se'),
            ResidualBlock(256, 256, stride=1, attention='se'),
            CBAM(256)
        )

        # Stage 4: High-level features (256 -> 512)
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, attention='cbam'),
            ResidualBlock(512, 512, stride=1, attention='cbam')
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # SRM preprocessing
        x = self.srm_layer(x)

        # Initial feature extraction with TLU activation
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = torch.tanh(x)  # TLU-like activation for steganalysis

        # Progressive feature extraction
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


# Example usage
if __name__ == "__main__":
    model = AttentionZhuNet(num_classes=2)

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
