# -*- coding: utf-8 -*-
"""D121-LowFE: the low-level patch feature extractor (paper Sec. 3.1, FGRE).

The paper truncates an ImageNet pre-trained DenseNet-121 to its first blocks
("initial 7x7 convolution included"), global-average-pools the selected
feature maps, flattens and concatenates them into the node features
``X = {f_1 ... f_N}, f_i in R^K`` (Eq. 1).

The original research pipeline exported the Keras ``DenseNet121`` up to

* ``conv2_block2_concat``  -> output of the 2nd dense layer of dense block 1
  (64 + 2x32 = **128** channels, 1/2 resolution), and
* ``conv3_block12_concat`` -> output of the final layer of dense block 2
  (128 + 12x32 = **512** channels, 1/4 resolution),

giving K = 512 + 128 = **640** node features per 40x40 landmark patch.  This
module reproduces that exact extractor with a torchvision DenseNet-121 so the
whole pipeline stays inside PyTorch (the concatenation order ``[512, 128]``
matches the Keras implementation that produced all released graph files).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class DenseNet121LowFE(nn.Module):
    """D121-LowFE (frozen front-end; Eq. 1 in the paper)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        features = torchvision.models.densenet121(weights=weights).features

        # conv0 (7x7, included per the paper) + norm + relu + 3x3 max-pool
        self.stem = nn.Sequential(features.conv0, features.norm0, features.relu0, features.pool0)
        # Dense block 1: only its first two dense layers are tapped (128 ch).
        self.block1 = features.denseblock1.layers[:2]
        self.transition1 = features.transition1
        # Dense block 2: all 12 dense layers are tapped (512 ch).
        self.block2 = features.denseblock2

    @torch.no_grad()
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """ patches: ``[B, 3, H, W]`` raw RGB in ``[0, 255]`` (e.g. 40x40 patches).

        Returns node features ``[B, 640]`` (GAP(block2) 512 | GAP(block1[:2]) 128,
        the exact concatenation order of the original Keras feature extractor).
        """
        x = (patches / 127.5) - 1.0          # Keras densenet.preprocess_input (mode='tf')
        x = self.stem(x)                     # [B, 64, H/2, W/2]
        for layer in self.block1:
            x = layer(x)                     # [B, 128, H/2, W/2]
        low = x
        x = self.transition1(x)
        x = self.block2(x)                   # [B, 512, H/4, W/4]
        pooled = torch.cat([x.mean(dim=(2, 3)), low.mean(dim=(2, 3))], dim=1)
        return pooled


__all__ = ["DenseNet121LowFE"]