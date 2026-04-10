"""Self/cross attention and transformer blocks.

MLX SD `unet.py::TransformerBlock`을 베이스로 LayerDiffuse의
`CrossFrameTransformerBlock`을 추가한다.

참조:
- ml-explore/mlx-examples/.../unet.py (TransformerBlock)
- shitagaki-lab/see-through/common/modules/layerdiffuse/transformer3d.py
  (CrossFrameTransformerBlock, Transformer3DModel)
- diffusers.models.attention.BasicTransformerBlock
- diffusers.models.attention.TemporalBasicTransformerBlock
"""

from __future__ import annotations

from typing import Any

MLXArray = Any  # see mlx_ops/embeddings.py for rationale


def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError as e:
        raise RuntimeError("MLX is required. pip install mlx on Apple Silicon.") from e


class TransformerBlock:
    """Self-attention + cross-attention + FF (BasicTransformerBlock).

    MLX SD 예제와 동일한 시그니처:
        __init__(model_dims, num_heads, hidden_dims=None, memory_dims=None)

    - self.norm1 + self.attn1 (self-attention)
    - self.norm2 + self.attn2 (cross-attention, memory_dims가 주어지면)
    - self.norm3 + self.ff (feed-forward)
    """

    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        hidden_dims: int | None = None,
        memory_dims: int | None = None,
    ) -> None:
        _require_mlx()
        self.model_dims = model_dims
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims or model_dims * 4
        self.memory_dims = memory_dims
        # TODO(stage1):
        #   import mlx.nn as nn
        #   self.norm1 = nn.LayerNorm(model_dims)
        #   self.attn1 = nn.MultiHeadAttention(model_dims, num_heads=num_heads)
        #   if memory_dims is not None:
        #       self.norm2 = nn.LayerNorm(model_dims)
        #       self.attn2 = nn.MultiHeadAttention(
        #           model_dims, num_heads=num_heads,
        #           key_input_dims=memory_dims,
        #       )
        #   self.norm3 = nn.LayerNorm(model_dims)
        #   self.ff = GeGlu(model_dims, self.hidden_dims)

    def __call__(
        self,
        x: MLXArray,
        encoder_x: MLXArray | None = None,
        attn_mask: MLXArray | None = None,
        encoder_attn_mask: MLXArray | None = None,
    ) -> MLXArray:
        """x: `(B, N, model_dims)` → same shape."""
        raise NotImplementedError(
            "TransformerBlock forward pending — see wiki/see-through-porting.md"
        )


class CrossFrameTransformerBlock:
    """See-Through 특유: 프레임(레이어)들 간 self-attention.

    표준 self-attention은 단일 프레임 내 공간적 self-attn만 수행.
    CrossFrameTransformerBlock은 프레임 축(레이어 축)으로도 attention을
    수행해 레이어들 간 시각적 일관성을 보장한다.

    원본 구조 (transformer3d.py):
        hidden_states: (batch, seq_length, num_frames, channels)
        → permute to (batch, num_frames, seq_length, channels)
        → apply temporal self-attention along num_frames dim
    """

    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        num_frames: int,
    ) -> None:
        _require_mlx()
        self.model_dims = model_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        # TODO(stage1): temporal attention layer

    def __call__(self, x: MLXArray) -> MLXArray:
        """x: `(B, seq_len, num_frames, model_dims)` → same shape."""
        raise NotImplementedError(
            "CrossFrameTransformerBlock forward pending — "
            "see wiki/see-through-porting.md"
        )
