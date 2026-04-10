"""Marigold MLX 모델 스켈레톤.

원본: https://github.com/prs-eth/marigold (Apache 2.0)
Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation.

See-Through에서는 fine-tuned Marigold를 사용해
레이어별 pseudo-depth를 추정하고, 이를 drawing order 결정에 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MarigoldMLX:
    """Marigold depth estimation MLX 래퍼 (미구현)."""

    weights_path: Path
    dtype: str = "bfloat16"
    _unet: Any = field(default=None, init=False, repr=False)
    _vae: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Marigold MLX 가중치 로드.

        Raises:
            NotImplementedError: MLX 포팅 미완료.
        """
        raise NotImplementedError(
            "Marigold MLX port pending — see wiki/see-through-porting.md"
        )

    def predict_depth(self, image: object) -> object:
        """단일 이미지에서 depth map 추정.

        Args:
            image: 입력 이미지 (MLX array).

        Returns:
            HxW float depth map (MLX array).

        Raises:
            NotImplementedError: MLX 포팅 미완료.
        """
        raise NotImplementedError(
            "Marigold inference pending — see wiki/see-through-porting.md"
        )
