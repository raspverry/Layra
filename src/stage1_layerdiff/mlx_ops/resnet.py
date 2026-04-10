"""ResnetBlock2D — MLX SD 예제와 동일 시그니처.

원본: ml-explore/mlx-examples/.../unet.py::ResnetBlock2D
"""

from __future__ import annotations

from typing import Any

MLXArray = Any  # see mlx_ops/embeddings.py for rationale


def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError as e:
        raise RuntimeError("MLX is required. pip install mlx on Apple Silicon.") from e


class ResnetBlock2D:
    """GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv residual block.

    Signature matches MLX SD:
        __init__(in_channels, out_channels=None, groups=32, temb_channels=None)

    - self.norm1, self.conv1
    - if temb_channels: self.time_emb_proj (Linear, temb_channels → out)
    - self.norm2, self.conv2
    - self.conv_shortcut if in ≠ out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        groups: int = 32,
        temb_channels: int | None = None,
    ) -> None:
        _require_mlx()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.groups = groups
        self.temb_channels = temb_channels
        # TODO(stage1):
        #   import mlx.nn as nn
        #   self.norm1 = nn.GroupNorm(groups, in_channels)
        #   self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        #   if temb_channels:
        #       self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)
        #   self.norm2 = nn.GroupNorm(groups, self.out_channels)
        #   self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        #   if in_channels != self.out_channels:
        #       self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, 1)

    def __call__(
        self,
        x: MLXArray,
        temb: MLXArray | None = None,
    ) -> MLXArray:
        """x: `(B, C_in, H, W)` → `(B, C_out, H, W)`."""
        raise NotImplementedError(
            "ResnetBlock2D forward pending — see wiki/see-through-porting.md"
        )
