"""Transformer3DModel + UNetBlock2D.

LayerDiffuse의 `transformer3d.py::Transformer3DModel`을 MLX로 이식.
SD의 `Transformer2DModel`(MLX SD `Transformer2D`)에 프레임 축
(num_frames=23)을 추가한 형태.

UNetBlock2D는 MLX SD와 같은 시그니처 (down/up 블록).
"""

from __future__ import annotations

from typing import Any

from src.stage1_layerdiff.mlx_ops.attention import (
    CrossFrameTransformerBlock,
    TransformerBlock,
)
from src.stage1_layerdiff.mlx_ops.resnet import ResnetBlock2D

MLXArray = Any  # see mlx_ops/embeddings.py for rationale


def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError as e:
        raise RuntimeError("MLX is required. pip install mlx on Apple Silicon.") from e


class Transformer3DModel:
    """Frame-aware Transformer stack.

    Input: `(B, C, num_frames, H, W)` — latent with layer axis.

    Pipeline (from transformer3d.py):
        1. Reshape to (B * num_frames, C, H, W) for spatial attention
        2. Apply TransformerBlock(s) (self + cross attention)
        3. Reshape to (B, seq, num_frames, C) for cross-frame attention
        4. Apply CrossFrameTransformerBlock(s)
        5. Reshape back and residual-add
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        num_frames: int,
        norm_num_groups: int = 32,
        use_cross_frame_attention: bool = True,
    ) -> None:
        _require_mlx()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames
        self.norm_num_groups = norm_num_groups
        self.use_cross_frame_attention = use_cross_frame_attention

        inner_dim = num_attention_heads * attention_head_dim
        self._inner_dim = inner_dim

        # TODO(stage1): construct:
        #   self.norm = nn.GroupNorm(norm_num_groups, in_channels)
        #   self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        #   self.transformer_blocks = [
        #       TransformerBlock(inner_dim, num_attention_heads, memory_dims=cross_attention_dim)
        #       for _ in range(num_layers)
        #   ]
        #   if use_cross_frame_attention:
        #       self.cross_frame_blocks = [
        #           CrossFrameTransformerBlock(inner_dim, num_attention_heads, num_frames)
        #           for _ in range(num_layers)
        #       ]
        #   self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1)

        # placeholders to keep type checkers happy
        self.transformer_blocks: list[TransformerBlock] = []
        self.cross_frame_blocks: list[CrossFrameTransformerBlock] = []

    def __call__(
        self,
        x: MLXArray,
        encoder_x: MLXArray | None = None,
        attn_mask: MLXArray | None = None,
        encoder_attn_mask: MLXArray | None = None,
    ) -> MLXArray:
        """x: `(B, C, num_frames, H, W)` → same shape."""
        raise NotImplementedError(
            "Transformer3DModel forward pending — see wiki/see-through-porting.md"
        )


class UNetBlock2D:
    """SDXL UNet down/up block with optional cross-attention.

    MLX SD `UNetBlock2D` + LayerDiffuse의 `num_frames` 전달 추가.
    transformer_layers_per_block == 0 이면 attention 없는 pure-resnet 블록
    (SDXL의 첫 스테이지).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        prev_out_channels: int | None = None,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 2048,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        add_upsample: bool = True,
        add_cross_attention: bool = True,
        num_frames: int = 23,
    ) -> None:
        _require_mlx()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.prev_out_channels = prev_out_channels
        self.num_layers = num_layers
        self.transformer_layers_per_block = transformer_layers_per_block
        self.num_attention_heads = num_attention_heads
        self.cross_attention_dim = cross_attention_dim
        self.resnet_groups = resnet_groups
        self.add_downsample = add_downsample
        self.add_upsample = add_upsample
        self.add_cross_attention = add_cross_attention
        self.num_frames = num_frames

        # TODO(stage1): construct ResnetBlock2D + Transformer3DModel lists,
        # optional downsample/upsample convs
        self.resnets: list[ResnetBlock2D] = []
        self.attentions: list[Transformer3DModel] = []

    def __call__(
        self,
        x: MLXArray,
        temb: MLXArray,
        encoder_x: MLXArray | None = None,
        attn_mask: MLXArray | None = None,
        encoder_attn_mask: MLXArray | None = None,
        residual_hidden_states: list[MLXArray] | None = None,
    ) -> tuple[MLXArray, list[MLXArray]]:
        """Returns (output, residual_hidden_states_for_upsample)."""
        raise NotImplementedError(
            "UNetBlock2D forward pending — see wiki/see-through-porting.md"
        )
