"""Neck YOLO26n, dipindahkan dari notebook penelitian asli."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastNormalizedFusion(nn.Module):
    """Fast normalized learnable weighted fusion."""

    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def normalized_weights(self):
        weights = F.relu(self.weights)
        return weights / (weights.sum() + self.eps)

    def forward(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Jumlah input feature tidak sesuai dengan fusion weights.")

        weights = self.normalized_weights()
        output = torch.zeros_like(inputs[0])

        for index, feature in enumerate(inputs):
            output = output + weights[index] * feature

        return output


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SeparableConvBNAct(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNNeck(nn.Module):
    """
    Satu BiFPN block untuk YOLO26n.

    Backbone input:
        P3: 128 channels, stride 8
        P4: 128 channels, stride 16
        P5: 256 channels, stride 32

    Internal fusion:
        128 channels untuk seluruh level.

    Detect output:
        P3: 64 channels
        P4: 128 channels
        P5: 256 channels
    """

    def __init__(
        self,
        in_channels=(128, 128, 256),
        fusion_channels=128,
        out_channels=(64, 128, 256),
    ):
        super().__init__()

        in_p3, in_p4, in_p5 = in_channels
        out_p3, out_p4, out_p5 = out_channels
        c = fusion_channels

        # Input alignment
        self.p3_in = ConvBNAct(in_p3, c, 1)
        self.p4_in = ConvBNAct(in_p4, c, 1)
        self.p5_in = ConvBNAct(in_p5, c, 1)

        # Top-down
        self.fuse_p4_td = FastNormalizedFusion(2)
        self.conv_p4_td = SeparableConvBNAct(c)

        self.fuse_p3_td = FastNormalizedFusion(2)
        self.conv_p3_td = SeparableConvBNAct(c)

        # Bottom-up
        self.downsample_p3 = nn.MaxPool2d(3, 2, 1)
        self.fuse_p4_out = FastNormalizedFusion(3)
        self.conv_p4_out = SeparableConvBNAct(c)

        self.downsample_p4 = nn.MaxPool2d(3, 2, 1)
        self.fuse_p5_out = FastNormalizedFusion(2)
        self.conv_p5_out = SeparableConvBNAct(c)

        # Restore Detect channel dimensions
        self.p3_out = ConvBNAct(c, out_p3, 1)
        self.p4_out = ConvBNAct(c, out_p4, 1)
        self.p5_out = ConvBNAct(c, out_p5, 1)

    def forward(self, features):
        if len(features) != 3:
            raise ValueError("BiFPNNeck membutuhkan P3, P4, dan P5.")

        p3_raw, p4_raw, p5_raw = features

        p3 = self.p3_in(p3_raw)
        p4 = self.p4_in(p4_raw)
        p5 = self.p5_in(p5_raw)

        # Top-down
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4_td = self.conv_p4_td(self.fuse_p4_td([p4, p5_up]))

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        p3_td = self.conv_p3_td(self.fuse_p3_td([p3, p4_up]))

        # Bottom-up
        p3_down = self.downsample_p3(p3_td)
        p4_bu = self.conv_p4_out(self.fuse_p4_out([p4, p4_td, p3_down]))

        p4_down = self.downsample_p4(p4_bu)
        p5_bu = self.conv_p5_out(self.fuse_p5_out([p5, p4_down]))

        return [
            self.p3_out(p3_td),
            self.p4_out(p4_bu),
            self.p5_out(p5_bu),
        ]


class FeatureSelect(nn.Module):
    """Ambil satu tensor dari list output BiFPN."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, features):
        return features[self.index]
