# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Ported and restructured for yaslowfast: binary classification, pure PyTorch,
# two tensor inputs, optional adaptive pooling. See THIRD_PARTY_NOTICES.md and licenses/.
"""Binary SlowFast-R50 with a model definition ported and restructured in PyTorch.

The PyTorchVideo ``slowfast_r50`` model definition was ported into this local
implementation. PyTorchVideo, torchvision, and fvcore are not dependencies for
execution, development, tests, or export. Copyright and Apache 2.0 notices are
retained to document the origin and license of the ported code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .labels import NUM_CLASSES

if TYPE_CHECKING:
    from .config import ModelConfig


def pack_pathways(frames: Tensor, alpha: int = 4) -> tuple[Tensor, Tensor]:
    """Pack RGB C,T,H,W or B,C,T,H,W frames using the reference linspace rule.

    For 32 frames this selects [0, 4, 8, 13, 17, 22, 26, 31], rather than ::4.
    """
    if frames.ndim not in (4, 5):
        raise ValueError("frames must have shape C,T,H,W or B,C,T,H,W")
    if alpha < 1:
        raise ValueError("alpha must be positive")
    temporal_dim = frames.ndim - 3
    length = frames.shape[temporal_dim]
    if length < alpha or length % alpha:
        raise ValueError("frame count must be a positive multiple of alpha")
    indices = torch.linspace(0, length - 1, length // alpha, device=frames.device).long()
    return torch.index_select(frames, temporal_dim, indices), frames


def _norm(channels: int) -> nn.BatchNorm3d:
    return nn.BatchNorm3d(channels, eps=1e-5, momentum=0.1)


class _Stem(nn.Module):
    def __init__(self, channels: int, temporal_kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            3,
            channels,
            (temporal_kernel, 7, 7),
            stride=(1, 2, 2),
            padding=(temporal_kernel // 2, 3, 3),
            bias=False,
        )
        self.norm = _norm(channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.activation(self.norm(self.conv(x))))


class _Bottleneck(nn.Module):
    def __init__(
        self,
        channels_in: int,
        inner: int,
        channels_out: int,
        temporal_kernel: int,
        spatial_stride: int,
    ) -> None:
        super().__init__()
        self.conv_a = nn.Conv3d(
            channels_in,
            inner,
            (temporal_kernel, 1, 1),
            padding=(temporal_kernel // 2, 0, 0),
            bias=False,
        )
        self.norm_a = _norm(inner)
        self.act_a = nn.ReLU()
        self.conv_b = nn.Conv3d(
            inner,
            inner,
            (1, 3, 3),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
            bias=False,
        )
        self.norm_b = _norm(inner)
        self.act_b = nn.ReLU()
        self.conv_c = nn.Conv3d(inner, channels_out, 1, bias=False)
        self.norm_c = _norm(channels_out)
        self.norm_c.block_final_bn = True

    def forward(self, x: Tensor) -> Tensor:
        x = self.act_a(self.norm_a(self.conv_a(x)))
        x = self.act_b(self.norm_b(self.conv_b(x)))
        return self.norm_c(self.conv_c(x))


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        inner: int,
        channels_out: int,
        temporal_kernel: int,
        spatial_stride: int,
    ) -> None:
        super().__init__()
        if channels_in != channels_out or spatial_stride != 1:
            self.branch1_conv = nn.Conv3d(
                channels_in,
                channels_out,
                1,
                stride=(1, spatial_stride, spatial_stride),
                bias=False,
            )
            self.branch1_norm = _norm(channels_out)
        else:
            self.branch1_conv = None
            self.branch1_norm = None
        self.branch2 = _Bottleneck(
            channels_in, inner, channels_out, temporal_kernel, spatial_stride
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        if self.branch1_conv is not None:
            shortcut = self.branch1_norm(self.branch1_conv(shortcut))
        return self.activation(shortcut + self.branch2(x))


class _Stage(nn.Module):
    def __init__(
        self,
        depth: int,
        channels_in: int,
        inner: int,
        channels_out: int,
        temporal_kernel: int,
        spatial_stride: int,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                _ResidualBlock(
                    channels_in if index == 0 else channels_out,
                    inner,
                    channels_out,
                    temporal_kernel,
                    spatial_stride if index == 0 else 1,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.res_blocks:
            x = block(x)
        return x


class _Fusion(nn.Module):
    def __init__(self, fast_channels: int) -> None:
        super().__init__()
        self.conv_fast_to_slow = nn.Conv3d(
            fast_channels,
            fast_channels * 2,
            (7, 1, 1),
            stride=(4, 1, 1),
            padding=(3, 0, 0),
            bias=False,
        )
        self.norm = _norm(fast_channels * 2)
        self.activation = nn.ReLU()

    def forward(self, slow: Tensor, fast: Tensor) -> tuple[Tensor, Tensor]:
        lateral = self.activation(self.norm(self.conv_fast_to_slow(fast)))
        return torch.cat((slow, lateral), dim=1), fast


class _PathwayBlock(nn.Module):
    def __init__(
        self, slow: nn.Module, fast: nn.Module, fast_channels: int, fuse: bool = True
    ) -> None:
        super().__init__()
        self.multipathway_blocks = nn.ModuleList([slow, fast])
        self.multipathway_fusion = _Fusion(fast_channels) if fuse else None

    def forward(self, slow: Tensor, fast: Tensor) -> tuple[Tensor, Tensor]:
        slow = self.multipathway_blocks[0](slow)
        fast = self.multipathway_blocks[1](fast)
        if self.multipathway_fusion is not None:
            slow, fast = self.multipathway_fusion(slow, fast)
        return slow, fast


class _PoolConcat(nn.Module):
    def __init__(self, pooling: str) -> None:
        super().__init__()
        if pooling == "adaptive":
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool3d(1), nn.AdaptiveAvgPool3d(1)])
        else:
            self.pool = nn.ModuleList(
                [
                    nn.AvgPool3d((8, 7, 7), stride=1),
                    nn.AvgPool3d((32, 7, 7), stride=1),
                ]
            )

    def forward(self, slow: Tensor, fast: Tensor) -> Tensor:
        return torch.cat((self.pool[0](slow), self.pool[1](fast)), dim=1)


class _Head(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.proj = nn.Linear(features, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x).permute(0, 2, 3, 4, 1)
        x = self.proj(x).permute(0, 4, 1, 2, 3)
        return x.mean(dim=(2, 3, 4))


class SlowFast(nn.Module):
    """Binary SlowFast-R50 with a ported and restructured local model definition.

    The model definition originates from PyTorchVideo's ``slowfast_r50`` and is
    implemented here using PyTorch, without a dependency on PyTorchVideo,
    torchvision, or fvcore. Weights are initialized from scratch. The fixed
    two-output head returns logits of shape ``[batch, 2]`` for class IDs 0 and 1;
    use cross entropy for training and softmax for class probabilities.

    ``width_per_group`` and ``stage_depths`` also support tiny smoke-test models.
    The reference R50 recipe always uses width 64 and depths (3, 4, 6, 3).
    Adaptive pooling equals the reference head at 32/8 x 224 x 224; larger inputs
    differ because the reference uses sliding fixed-kernel average pooling.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        pooling: str = "adaptive",
        width_per_group: int = 64,
        stage_depths: tuple[int, int, int, int] = (3, 4, 6, 3),
    ) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes != NUM_CLASSES:
            raise ValueError("binary classification requires num_classes=2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if pooling not in ("adaptive", "reference"):
            raise ValueError("pooling must be adaptive or reference")
        if width_per_group < 8 or width_per_group % 8:
            raise ValueError("width_per_group must be a positive multiple of 8")
        if len(stage_depths) != 4 or any(depth < 1 for depth in stage_depths):
            raise ValueError("stage_depths must contain four positive integers")
        self.num_classes = num_classes
        self.pooling = pooling
        self.width_per_group = width_per_group
        self.stage_depths = tuple(stage_depths)
        width = width_per_group
        blocks: list[nn.Module] = [_PathwayBlock(_Stem(width, 1), _Stem(width // 8, 5), width // 8)]
        channels_in = width
        for index, depth in enumerate(stage_depths):
            inner = width * 2**index
            channels_out = inner * 4
            stride = 1 if index == 0 else 2
            blocks.append(
                _PathwayBlock(
                    _Stage(
                        depth,
                        channels_in + channels_in // 4,
                        inner,
                        channels_out,
                        1 if index < 2 else 3,
                        stride,
                    ),
                    _Stage(depth, channels_in // 8, inner // 8, channels_out // 8, 3, stride),
                    channels_out // 8,
                    fuse=index < 3,
                )
            )
            channels_in = channels_out
        blocks.extend(
            [
                _PoolConcat(pooling),
                _Head(channels_in + channels_in // 8, dropout),
            ]
        )
        self.blocks = nn.ModuleList(blocks)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """MSRA convolution initialization and zero final residual BN scales."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(
                    module.weight, 0 if getattr(module, "block_final_bn", False) else 1
                )
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, slow: Tensor, fast: Tensor) -> Tensor:
        if slow.ndim != 5 or fast.ndim != 5:
            raise ValueError("inputs must have shape B,C,T,H,W")
        if slow.shape[1] != 3 or fast.shape[1] != 3:
            raise ValueError("SlowFast expects RGB inputs")
        if slow.shape[0] != fast.shape[0] or slow.shape[3:] != fast.shape[3:]:
            raise ValueError("pathways must have the same batch and spatial dimensions")
        if fast.shape[2] != 4 * slow.shape[2] or slow.shape[2] < 1:
            raise ValueError("Fast temporal length must be four times Slow temporal length")
        if self.pooling == "reference" and (slow.shape[2] != 8 or fast.shape[2] != 32):
            raise ValueError("reference pooling requires 32 Fast and 8 Slow frames")
        for index in range(5):
            slow, fast = self.blocks[index](slow, fast)
        x = self.blocks[5](slow, fast)
        return self.blocks[6](x)


def build_model(model_config: ModelConfig, num_classes: int = NUM_CLASSES) -> SlowFast:
    """Construct the local binary model with random weights and no upstream dependency."""
    if model_config.alpha != 4:
        raise ValueError("this SlowFast-R50 implementation requires alpha=4")
    return SlowFast(
        num_classes=num_classes,
        dropout=model_config.dropout,
        pooling=model_config.pooling,
        width_per_group=model_config.width_per_group,
        stage_depths=tuple(model_config.stage_depths),
    )
