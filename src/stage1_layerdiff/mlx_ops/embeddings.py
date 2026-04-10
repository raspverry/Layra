"""Timestep + SDXL addition embeddings.

MLX SD `unet.py::TimestepEmbedding`과 SDXL의 `add_text_embeds` / `time_ids`
처리 로직의 MLX 이식.

포팅 참조:
- ml-explore/mlx-examples/stable_diffusion/stable_diffusion/unet.py
- diffusers/models/embeddings.py (Timesteps, TimestepEmbedding)
"""

from __future__ import annotations

from typing import Any

# mlx is only installed on Apple Silicon (Darwin/arm64). On other platforms
# this alias collapses to Any so mypy/ruff can still check the file.
# At runtime on Mac we'd use mlx.core.array, but keeping the alias as Any
# avoids introducing a conditional-type-alias branch mypy can't follow.
MLXArray = Any


def timestep_embedding(
    timesteps: MLXArray,
    dim: int,
    max_period: float = 10000.0,
) -> MLXArray:
    """Sinusoidal positional embedding for diffusion timesteps.

    `diffusers.models.embeddings.Timesteps`의 MLX 버전.

    Args:
        timesteps: `(batch,)` int32 MLX 배열.
        dim: 출력 임베딩 차원 (짝수여야 함).
        max_period: sinusoid 주기.

    Returns:
        `(batch, dim)` float MLX 배열.
    """
    # TODO(stage1): MLX 구현
    # ref (PyTorch):
    #   half = dim // 2
    #   freqs = torch.exp(
    #       -math.log(max_period) * torch.arange(half) / half
    #   )
    #   args = timesteps[:, None] * freqs[None]
    #   emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    raise NotImplementedError(
        "timestep_embedding MLX port pending — see wiki/see-through-porting.md"
    )


def _require_mlx() -> None:
    """MLX가 설치되었는지 확인하고 안 되면 명확히 실패."""
    try:
        import mlx  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MLX is required for Stage 1 inference. "
            "Install on Apple Silicon with: pip install mlx"
        ) from e


class TimestepEmbedding:
    """Timestep → hidden projection.

    PyTorch:
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
    """

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        _require_mlx()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        # TODO(stage1):
        #   import mlx.nn as nn
        #   self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        #   self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x: MLXArray) -> MLXArray:
        """x: `(batch, in_channels)` → `(batch, time_embed_dim)`."""
        raise NotImplementedError(
            "TimestepEmbedding forward pending — see wiki/see-through-porting.md"
        )


class SDXLAdditionEmbedding:
    """SDXL micro-conditioning embedding.

    SDXL은 `add_text_embeds` (pooled CLIP-L) + `add_time_ids`
    (original_size, crops_coords_top_left, target_size)를
    hidden에 결합한다.

    원본 (diffusers.models.unet_2d_condition.UNet2DConditionModel):
        aug_emb = self.add_embedding(
            torch.cat([add_text_embeds, add_time_embed], dim=-1)
        )
        emb = emb + aug_emb
    """

    def __init__(
        self,
        addition_time_embed_dim: int,
        projection_dim: int,
        time_embed_dim: int,
    ) -> None:
        _require_mlx()
        self.addition_time_embed_dim = addition_time_embed_dim
        self.projection_dim = projection_dim
        self.time_embed_dim = time_embed_dim
        # TODO(stage1): linear projection from (projection_dim + 6*addition_time_embed_dim)
        #               → time_embed_dim

    def __call__(
        self,
        add_text_embeds: MLXArray,
        add_time_ids: MLXArray,
    ) -> MLXArray:
        """Returns `(batch, time_embed_dim)` aug embedding."""
        raise NotImplementedError(
            "SDXLAdditionEmbedding forward pending — see wiki/see-through-porting.md"
        )
