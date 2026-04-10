"""Normalization layers for LayerDiffuse.

See-Through의 `transformer3d.py`가 사용하는 `AdaLayerNormSingle`은
PixArt 스타일의 time-conditioned LayerNorm이다.

참조: diffusers.models.normalization.AdaLayerNormSingle
"""

from __future__ import annotations

from typing import Any

MLXArray = Any  # see mlx_ops/embeddings.py for rationale


def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MLX is required. pip install mlx on Apple Silicon."
        ) from e


class AdaLayerNormSingle:
    """Time-conditioned LayerNorm 단일 버전 (PixArt / LayerDiffuse).

    원본 (diffusers.models.normalization.AdaLayerNormSingle):
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim, size_emb_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    forward는 (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    6-tuple을 반환하여 transformer block이 modulation에 사용한다.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_additional_conditions: bool = False,
    ) -> None:
        _require_mlx()
        self.embedding_dim = embedding_dim
        self.use_additional_conditions = use_additional_conditions
        # TODO(stage1):
        #   import mlx.nn as nn
        #   self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def __call__(
        self,
        timestep: MLXArray,
        added_cond_kwargs: dict[str, MLXArray] | None = None,
        batch_size: int | None = None,
        hidden_dtype: str | None = None,
    ) -> tuple[MLXArray, MLXArray, MLXArray]:
        """Returns (timestep_embed, embedded_timestep, mod_params).

        mod_params: shape `(batch, 6 * embedding_dim)`.
        """
        raise NotImplementedError(
            "AdaLayerNormSingle forward pending — "
            "see wiki/see-through-porting.md"
        )
