"""MarigoldMLX — depth estimation wrapper.

원본: prs-eth/marigold (Apache 2.0)
See-Through는 이 모델을 fine-tune해 layer별 pseudo-depth를 추정한다.
drawing order 결정에 사용.

참조: common/modules/marigold/marigold_depth_pipeline.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger
from src.stage1_layerdiff.configs import DiffusionConfig, MarigoldConfig
from src.stage1_layerdiff.schedulers import DPMSolverMultistep

if TYPE_CHECKING:
    import numpy as np

MLXArray = Any  # see mlx_ops/embeddings.py for rationale

logger = get_logger(__name__)


@dataclass(slots=True)
class MarigoldMLX:
    """Marigold depth MLX 래퍼."""

    weights_path: Path
    model_config: MarigoldConfig = field(default_factory=MarigoldConfig)
    diffusion_config: DiffusionConfig = field(default_factory=DiffusionConfig)
    dtype: str = "bfloat16"

    _unet: Any = field(default=None, init=False, repr=False)
    _vae: Any = field(default=None, init=False, repr=False)
    _text_encoder: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Load Marigold components (UNet, VAE, text encoder).

        Marigold는 SD 1.5 기반이라 MLX SD 예제의 UNetModel을 거의
        그대로 사용할 수 있다 (transformer3d는 불필요).
        """
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "MLX is required. pip install mlx on Apple Silicon."
            ) from e
        # TODO(stage1): load SD 1.5-like UNet + VAE + text encoder
        raise NotImplementedError(
            "Marigold MLX load pending — see wiki/see-through-porting.md"
        )

    def predict_depth(
        self,
        image: np.ndarray,
        ensemble_size: int | None = None,
        num_inference_steps: int | None = None,
    ) -> np.ndarray:
        """Estimate depth map for a single image.

        Marigold는 여러 번 샘플링해 ensemble 후 평균을 반환한다.

        Args:
            image: HxWx3 uint8 numpy 배열.
            ensemble_size: 샘플링 반복 수 (None이면 config 값).
            num_inference_steps: 디퓨전 스텝 수 (None이면 config 값).

        Returns:
            HxW float32 depth map (0~1 정규화).
        """
        if self._unet is None:
            self.load()
        raise NotImplementedError(
            "Marigold depth inference pending — "
            "see wiki/see-through-porting.md"
        )

    def _build_sampler(self, num_inference_steps: int) -> DPMSolverMultistep:
        return DPMSolverMultistep(
            config=self.diffusion_config,
            num_inference_steps=num_inference_steps,
        )
