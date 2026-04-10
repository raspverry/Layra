"""LayerDiffuse MLX 모델 스켈레톤.

원본: https://github.com/lllyasviel/LayerDiffuse
See-Through: SDXL 기반 LayerDiffuse로 학습된 레이어 분해 모델.

포팅 전략:
    1. MLX Stable Diffusion 예제(mlx-examples/stable_diffusion)를 베이스로
       SDXL UNet 구조를 MLX로 재구현.
    2. LayerDiffuse 특유의 transparent layer head는 별도 모듈로 구성.
    3. PyTorch state_dict → MLX weights 변환은 weights.py 참조.

블로커:
    - BLOCKER-002: 커스텀 CUDA 커널 / xformers 사용 여부 검증 필요.
    - See-Through 원본 코드 정독 후 UNet 블록 단위로 포팅.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LayerDiffuseMLX:
    """LayerDiffuse MLX UNet 래퍼 (미구현)."""

    weights_path: Path
    dtype: str = "bfloat16"
    _unet: Any = field(default=None, init=False, repr=False)
    _vae: Any = field(default=None, init=False, repr=False)
    _text_encoder: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """MLX 가중치를 메모리에 올린다.

        Raises:
            NotImplementedError: MLX 포팅 미완료.
        """
        # TODO(stage1): mlx.core.load(self.weights_path / "unet.npz")
        # TODO(stage1): VAE, text encoder 로드
        raise NotImplementedError(
            "LayerDiffuse MLX port pending — see wiki/see-through-porting.md"
        )

    def sample(
        self,
        conditioning: object,
        num_steps: int,
        guidance_scale: float,
    ) -> object:
        """Diffusion sampling loop.

        Args:
            conditioning: 텍스트/이미지 conditioning (타입은 구현 시 확정).
            num_steps: DDIM step 수.
            guidance_scale: CFG scale.

        Returns:
            MLX array (sampled latent).

        Raises:
            NotImplementedError: MLX 포팅 미완료.
        """
        raise NotImplementedError(
            "LayerDiffuse sampling pending — see wiki/see-through-porting.md"
        )
