"""Neck YOLO26n, dipindahkan dari notebook penelitian asli."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
    ):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2 if kernel_size else 0

        self.block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (
                        "bn",
                        nn.BatchNorm2d(out_channels),
                    ),
                    (
                        "silu",
                        nn.SiLU(inplace=True),
                    ),
                ]
            )
        )

    def forward(self, x):
        return self.block(x)


class ResidualBasicBlock(nn.Module):
    """
    Residual block following the official YOLOv5AFPN implementation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        if in_channels != out_channels:
            raise ValueError("ResidualBasicBlock requires equal input/output channels.")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels,
            momentum=0.1,
        )
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(
            out_channels,
            momentum=0.1,
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        return self.act(out)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=2,
    ):
        super().__init__()

        self.proj = BasicConv(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.proj(x)
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )


class DownsampleX2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.down = BasicConv(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x):
        return self.down(x)


class DownsampleX4(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.down = BasicConv(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=4,
            padding=0,
        )

    def forward(self, x):
        return self.down(x)


class AdaptiveSpatialFusion2(nn.Module):
    """
    Two-input adaptive spatial fusion used in AFPN stage 1.
    """

    def __init__(
        self,
        inter_dim,
        compress_channels=8,
    ):
        super().__init__()

        self.weight_level_1 = BasicConv(
            inter_dim,
            compress_channels,
            kernel_size=1,
        )
        self.weight_level_2 = BasicConv(
            inter_dim,
            compress_channels,
            kernel_size=1,
        )
        self.weight_levels = nn.Conv2d(
            compress_channels * 2,
            2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv = BasicConv(
            inter_dim,
            inter_dim,
            kernel_size=3,
            stride=1,
        )

    def forward(
        self,
        input1,
        input2,
        return_weights=False,
    ):
        if input1.shape != input2.shape:
            raise RuntimeError(
                "AdaptiveSpatialFusion2 requires "
                f"equal shapes, got {input1.shape} vs {input2.shape}."
            )

        weight1 = self.weight_level_1(input1)
        weight2 = self.weight_level_2(input2)

        logits = self.weight_levels(
            torch.cat(
                [weight1, weight2],
                dim=1,
            )
        )

        weights = F.softmax(
            logits,
            dim=1,
        )

        fused = input1 * weights[:, 0:1] + input2 * weights[:, 1:2]

        out = self.conv(fused)

        if return_weights:
            return out, weights

        return out


class AdaptiveSpatialFusion3(nn.Module):
    """
    Three-input adaptive spatial fusion used in AFPN stage 2.
    """

    def __init__(
        self,
        inter_dim,
        compress_channels=8,
    ):
        super().__init__()

        self.weight_level_1 = BasicConv(
            inter_dim,
            compress_channels,
            kernel_size=1,
        )
        self.weight_level_2 = BasicConv(
            inter_dim,
            compress_channels,
            kernel_size=1,
        )
        self.weight_level_3 = BasicConv(
            inter_dim,
            compress_channels,
            kernel_size=1,
        )

        self.weight_levels = nn.Conv2d(
            compress_channels * 3,
            3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.conv = BasicConv(
            inter_dim,
            inter_dim,
            kernel_size=3,
            stride=1,
        )

    def forward(
        self,
        input1,
        input2,
        input3,
        return_weights=False,
    ):
        if not (input1.shape == input2.shape == input3.shape):
            raise RuntimeError("AdaptiveSpatialFusion3 requires equal input shapes.")

        weight1 = self.weight_level_1(input1)
        weight2 = self.weight_level_2(input2)
        weight3 = self.weight_level_3(input3)

        logits = self.weight_levels(
            torch.cat(
                [
                    weight1,
                    weight2,
                    weight3,
                ],
                dim=1,
            )
        )

        weights = F.softmax(
            logits,
            dim=1,
        )

        fused = (
            input1 * weights[:, 0:1]
            + input2 * weights[:, 1:2]
            + input3 * weights[:, 2:3]
        )

        out = self.conv(fused)

        if return_weights:
            return out, weights

        return out


def make_residual_stack(
    channels,
    num_blocks,
):
    return nn.Sequential(
        *[
            ResidualBasicBlock(
                channels,
                channels,
            )
            for _ in range(num_blocks)
        ]
    )


class AFPNScaleBlockBody(nn.Module):
    """
    Three-level asymptotic fusion body based on the official YOLOv5AFPN.

    Stage 1:
        P3 <-> P4 only.

    Stage 2:
        progressively incorporate P5 into the refined P3/P4 representations.
    """

    def __init__(
        self,
        channels=(32, 32, 64),
        compress_channels=8,
        residual_blocks=3,
    ):
        super().__init__()

        c3, c4, c5 = channels

        # Initial same-scale refinement
        self.pre_p3 = BasicConv(c3, c3, 1)
        self.pre_p4 = BasicConv(c4, c4, 1)
        self.pre_p5 = BasicConv(c5, c5, 1)

        # ----------------------------------------------------
        # Stage 1: fuse adjacent low/mid levels only
        # ----------------------------------------------------
        self.p4_to_p3_s1 = Upsample(c4, c3, scale_factor=2)
        self.p3_to_p4_s1 = DownsampleX2(c3, c4)

        self.fuse_p3_s1 = AdaptiveSpatialFusion2(
            c3,
            compress_channels,
        )
        self.fuse_p4_s1 = AdaptiveSpatialFusion2(
            c4,
            compress_channels,
        )

        self.refine_p3_s1 = make_residual_stack(
            c3,
            residual_blocks,
        )
        self.refine_p4_s1 = make_residual_stack(
            c4,
            residual_blocks,
        )

        # ----------------------------------------------------
        # Stage 2: incorporate high-level P5
        # ----------------------------------------------------
        self.p4_to_p3_s2 = Upsample(c4, c3, scale_factor=2)
        self.p5_to_p4_s2 = Upsample(c5, c4, scale_factor=2)
        self.p5_to_p3_s2 = Upsample(c5, c3, scale_factor=4)

        self.p3_to_p4_s2 = DownsampleX2(c3, c4)
        self.p3_to_p5_s2 = DownsampleX4(c3, c5)
        self.p4_to_p5_s2 = DownsampleX2(c4, c5)

        self.fuse_p3_s2 = AdaptiveSpatialFusion3(
            c3,
            compress_channels,
        )
        self.fuse_p4_s2 = AdaptiveSpatialFusion3(
            c4,
            compress_channels,
        )
        self.fuse_p5_s2 = AdaptiveSpatialFusion3(
            c5,
            compress_channels,
        )

        self.refine_p3_s2 = make_residual_stack(
            c3,
            residual_blocks,
        )
        self.refine_p4_s2 = make_residual_stack(
            c4,
            residual_blocks,
        )
        self.refine_p5_s2 = make_residual_stack(
            c5,
            residual_blocks,
        )

    def forward(
        self,
        features,
        return_weights=False,
    ):
        if len(features) != 3:
            raise ValueError("AFPNScaleBlockBody requires P3, P4, P5.")

        p3, p4, p5 = features

        p3 = self.pre_p3(p3)
        p4 = self.pre_p4(p4)
        p5 = self.pre_p5(p5)

        # ====================================================
        # Stage 1: only adjacent P3/P4
        # ====================================================
        if return_weights:
            s1_p3, w_s1_p3 = self.fuse_p3_s1(
                p3,
                self.p4_to_p3_s1(p4),
                return_weights=True,
            )

            s1_p4, w_s1_p4 = self.fuse_p4_s1(
                self.p3_to_p4_s1(p3),
                p4,
                return_weights=True,
            )
        else:
            s1_p3 = self.fuse_p3_s1(
                p3,
                self.p4_to_p3_s1(p4),
            )

            s1_p4 = self.fuse_p4_s1(
                self.p3_to_p4_s1(p3),
                p4,
            )

        p3_refined = self.refine_p3_s1(s1_p3)
        p4_refined = self.refine_p4_s1(s1_p4)

        # ====================================================
        # Stage 2: progressively incorporate P5
        # ====================================================
        if return_weights:
            s2_p3, w_s2_p3 = self.fuse_p3_s2(
                p3_refined,
                self.p4_to_p3_s2(p4_refined),
                self.p5_to_p3_s2(p5),
                return_weights=True,
            )

            s2_p4, w_s2_p4 = self.fuse_p4_s2(
                self.p3_to_p4_s2(p3_refined),
                p4_refined,
                self.p5_to_p4_s2(p5),
                return_weights=True,
            )

            s2_p5, w_s2_p5 = self.fuse_p5_s2(
                self.p3_to_p5_s2(p3_refined),
                self.p4_to_p5_s2(p4_refined),
                p5,
                return_weights=True,
            )
        else:
            s2_p3 = self.fuse_p3_s2(
                p3_refined,
                self.p4_to_p3_s2(p4_refined),
                self.p5_to_p3_s2(p5),
            )

            s2_p4 = self.fuse_p4_s2(
                self.p3_to_p4_s2(p3_refined),
                p4_refined,
                self.p5_to_p4_s2(p5),
            )

            s2_p5 = self.fuse_p5_s2(
                self.p3_to_p5_s2(p3_refined),
                self.p4_to_p5_s2(p4_refined),
                p5,
            )

        out_p3 = self.refine_p3_s2(s2_p3)
        out_p4 = self.refine_p4_s2(s2_p4)
        out_p5 = self.refine_p5_s2(s2_p5)

        if return_weights:
            return (
                [out_p3, out_p4, out_p5],
                {
                    "stage1_P3": w_s1_p3,
                    "stage1_P4": w_s1_p4,
                    "stage2_P3": w_s2_p3,
                    "stage2_P4": w_s2_p4,
                    "stage2_P5": w_s2_p5,
                },
            )

        return [
            out_p3,
            out_p4,
            out_p5,
        ]


class AFPNNeck(nn.Module):
    """
    Literature-grounded three-level AFPN adaptation for YOLO26n.

    Raw YOLO26n backbone input:
        P3: 128 channels, stride 8
        P4: 128 channels, stride 16
        P5: 256 channels, stride 32

    Official AFPN-style reduction ratio:
        /4 -> 32 / 32 / 64 internal channels

    Detect output:
        P3: 64 channels
        P4: 128 channels
        P5: 256 channels
    """

    def __init__(
        self,
        in_channels=(128, 128, 256),
        out_channels=(64, 128, 256),
        reduction_ratio=4,
        compress_channels=8,
        residual_blocks=3,
    ):
        super().__init__()

        if any(c % reduction_ratio != 0 for c in in_channels):
            raise ValueError("in_channels must be divisible by reduction_ratio.")

        internal = tuple(c // reduction_ratio for c in in_channels)

        self.in_channels = tuple(in_channels)
        self.out_channels = tuple(out_channels)
        self.internal_channels = internal
        self.reduction_ratio = reduction_ratio
        self.compress_channels = compress_channels
        self.residual_blocks = residual_blocks

        # Input compression: same design principle as official YOLOv5AFPN.
        self.reduce_p3 = BasicConv(
            in_channels[0],
            internal[0],
            kernel_size=1,
        )
        self.reduce_p4 = BasicConv(
            in_channels[1],
            internal[1],
            kernel_size=1,
        )
        self.reduce_p5 = BasicConv(
            in_channels[2],
            internal[2],
            kernel_size=1,
        )

        self.body = AFPNScaleBlockBody(
            channels=internal,
            compress_channels=compress_channels,
            residual_blocks=residual_blocks,
        )

        # Restore the unchanged YOLO26n Detect interface.
        self.out_p3 = BasicConv(
            internal[0],
            out_channels[0],
            kernel_size=1,
        )
        self.out_p4 = BasicConv(
            internal[1],
            out_channels[1],
            kernel_size=1,
        )
        self.out_p5 = BasicConv(
            internal[2],
            out_channels[2],
            kernel_size=1,
        )

        self._reset_parameters_official()

    def _reset_parameters_official(
        self,
    ):
        """
        Match the explicit initialization used by the official AFPN code.
        Determinism is provided by the seed set before module construction.
        """
        for module in self.modules():
            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.xavier_normal_(
                    module.weight,
                    gain=0.02,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.normal_(
                    module.weight,
                    mean=1.0,
                    std=0.02,
                )
                nn.init.constant_(
                    module.bias,
                    0.0,
                )

    def _reduce(self, features):
        if len(features) != 3:
            raise ValueError("AFPNNeck requires P3, P4, P5.")

        p3, p4, p5 = features

        return [
            self.reduce_p3(p3),
            self.reduce_p4(p4),
            self.reduce_p5(p5),
        ]

    def forward(self, features):
        reduced = self._reduce(features)

        p3, p4, p5 = self.body(reduced)

        return [
            self.out_p3(p3),
            self.out_p4(p4),
            self.out_p5(p5),
        ]

    def forward_with_weights(
        self,
        features,
    ):
        reduced = self._reduce(features)

        outputs, weights = self.body(
            reduced,
            return_weights=True,
        )

        p3, p4, p5 = outputs

        return (
            [
                self.out_p3(p3),
                self.out_p4(p4),
                self.out_p5(p5),
            ],
            weights,
        )


class FeatureSelect(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, features):
        return features[self.index]
