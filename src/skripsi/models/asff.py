"""Neck YOLO26n, dipindahkan dari notebook penelitian asli."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                c1,
                c2,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ASFFScale(nn.Module):
    """
    ASFF fusion untuk satu target scale.

    target_level:
        0 -> P3 / stride 8
        1 -> P4 / stride 16
        2 -> P5 / stride 32

    Bobot ASFF berbentuk [B, 3, H, W], sehingga setiap lokasi
    mempunyai alpha/beta/gamma sendiri untuk sumber P3/P4/P5.
    Bobot dibagi pada seluruh channel di lokasi yang sama.
    """

    def __init__(
        self,
        target_level,
        in_channels=(128, 128, 256),
        out_channels=(64, 128, 256),
        weight_channels=16,
    ):
        super().__init__()

        self.target_level = int(target_level)

        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels

        if self.target_level == 0:
            c = o3

            self.p3_resize = ConvBNAct(c3, c, 1)
            self.p4_resize = ConvBNAct(c4, c, 1)
            self.p5_resize = ConvBNAct(c5, c, 1)

        elif self.target_level == 1:
            c = o4

            # P3: 80x80 -> 40x40
            self.p3_resize = ConvBNAct(c3, c, kernel_size=3, stride=2)

            # P4 already 128x40x40
            self.p4_resize = nn.Identity()

            # P5: 256 -> 128, lalu 20x20 -> 40x40
            self.p5_resize = ConvBNAct(c5, c, 1)

        elif self.target_level == 2:
            c = o5

            # P3: 80x80 -> 40x40 -> 20x20
            self.p3_pre = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.p3_resize = ConvBNAct(c3, c, kernel_size=3, stride=2)

            # P4: 40x40 -> 20x20
            self.p4_resize = ConvBNAct(c4, c, kernel_size=3, stride=2)

            # P5 already 256x20x20
            self.p5_resize = nn.Identity()

        else:
            raise ValueError("target_level harus 0, 1, atau 2.")

        self.inter_channels = c

        # Compressed representations used to predict 3 spatial weights.
        self.weight_p3 = ConvBNAct(c, weight_channels, 1)
        self.weight_p4 = ConvBNAct(c, weight_channels, 1)
        self.weight_p5 = ConvBNAct(c, weight_channels, 1)

        self.weight_levels = nn.Conv2d(
            weight_channels * 3,
            3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Post-fusion refinement.
        self.expand = ConvBNAct(c, c, kernel_size=3, stride=1)

    def _resize_features(self, p3, p4, p5):
        if self.target_level == 0:
            x3 = self.p3_resize(p3)
            target_size = x3.shape[-2:]

            x4 = F.interpolate(
                self.p4_resize(p4),
                size=target_size,
                mode="nearest",
            )
            x5 = F.interpolate(
                self.p5_resize(p5),
                size=target_size,
                mode="nearest",
            )

        elif self.target_level == 1:
            x3 = self.p3_resize(p3)
            x4 = self.p4_resize(p4)
            target_size = x4.shape[-2:]

            x5 = F.interpolate(
                self.p5_resize(p5),
                size=target_size,
                mode="nearest",
            )

            if x3.shape[-2:] != target_size:
                raise RuntimeError("P3->P4 spatial alignment gagal.")

        else:
            x3 = self.p3_resize(self.p3_pre(p3))
            x4 = self.p4_resize(p4)
            x5 = self.p5_resize(p5)
            target_size = x5.shape[-2:]

            if x3.shape[-2:] != target_size:
                raise RuntimeError("P3->P5 spatial alignment gagal.")
            if x4.shape[-2:] != target_size:
                raise RuntimeError("P4->P5 spatial alignment gagal.")

        if not (x3.shape == x4.shape == x5.shape):
            raise RuntimeError(
                "ASFF aligned feature shapes tidak identik: "
                f"{x3.shape}, {x4.shape}, {x5.shape}"
            )

        return x3, x4, x5

    def forward(
        self,
        p3,
        p4,
        p5,
        return_weights=False,
    ):
        x3, x4, x5 = self._resize_features(p3, p4, p5)

        weight_features = torch.cat(
            [
                self.weight_p3(x3),
                self.weight_p4(x4),
                self.weight_p5(x5),
            ],
            dim=1,
        )

        logits = self.weight_levels(weight_features)
        weights = F.softmax(logits, dim=1)

        fused = x3 * weights[:, 0:1] + x4 * weights[:, 1:2] + x5 * weights[:, 2:3]

        output = self.expand(fused)

        if return_weights:
            return output, weights

        return output


class ASFFNeck(nn.Module):
    """
    Three-level ASFF neck adaptation for YOLO26n.

    Backbone input:
        P3: 128 x 80 x 80
        P4: 128 x 40 x 40
        P5: 256 x 20 x 20

    Detect output:
        P3:  64 x 80 x 80
        P4: 128 x 40 x 40
        P5: 256 x 20 x 20
    """

    def __init__(
        self,
        in_channels=(128, 128, 256),
        out_channels=(64, 128, 256),
        weight_channels=16,
    ):
        super().__init__()

        self.asff_p3 = ASFFScale(
            target_level=0,
            in_channels=in_channels,
            out_channels=out_channels,
            weight_channels=weight_channels,
        )
        self.asff_p4 = ASFFScale(
            target_level=1,
            in_channels=in_channels,
            out_channels=out_channels,
            weight_channels=weight_channels,
        )
        self.asff_p5 = ASFFScale(
            target_level=2,
            in_channels=in_channels,
            out_channels=out_channels,
            weight_channels=weight_channels,
        )

    def forward(self, features):
        if len(features) != 3:
            raise ValueError("ASFFNeck membutuhkan P3, P4, dan P5.")

        p3, p4, p5 = features

        return [
            self.asff_p3(p3, p4, p5),
            self.asff_p4(p3, p4, p5),
            self.asff_p5(p3, p4, p5),
        ]

    def forward_with_weights(self, features):
        if len(features) != 3:
            raise ValueError("ASFFNeck membutuhkan P3, P4, dan P5.")

        p3, p4, p5 = features

        out_p3, w_p3 = self.asff_p3(p3, p4, p5, return_weights=True)
        out_p4, w_p4 = self.asff_p4(p3, p4, p5, return_weights=True)
        out_p5, w_p5 = self.asff_p5(p3, p4, p5, return_weights=True)

        return (
            [out_p3, out_p4, out_p5],
            {
                "P3": w_p3,
                "P4": w_p4,
                "P5": w_p5,
            },
        )


class FeatureSelect(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, features):
        return features[self.index]
