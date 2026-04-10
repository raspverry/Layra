"""LayerDiffuseMLX — Stage 1의 SDXL+frame UNet 고수준 래퍼.

이 모듈은 `unet_frame.UNetFrameConditionModel`과
`vae.TransparentVAE`, SDXL 텍스트 인코더 두 개, 그리고
`schedulers.DPMSolverMultistep`를 묶어 `sample()` API로 제공한다.

맥북 도착 후 채울 TODO가 대부분이지만, 인터페이스는 확정되어 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger
from src.stage1_layerdiff.configs import (
    DiffusionConfig,
    SDXLTextEncoderConfig,
    TransparentVAEConfig,
    UNetFrameConditionConfig,
)
from src.stage1_layerdiff.schedulers import DPMSolverMultistep
from src.stage1_layerdiff.unet_frame import UNetFrameConditionModel
from src.stage1_layerdiff.vae import TransparentVAE

if TYPE_CHECKING:
    import numpy as np

MLXArray = Any  # see mlx_ops/embeddings.py for rationale

logger = get_logger(__name__)


@dataclass(slots=True)
class LayerDiffuseMLX:
    """LayerDiffuse MLX 고수준 파이프라인.

    구성요소:
        - UNetFrameConditionModel (SDXL + frame)
        - TransparentVAE (RGBA 출력)
        - Two CLIP text encoders (SDXL)
        - DPMSolverMultistep (DPM++ 2M SDE)
    """

    weights_path: Path
    unet_config: UNetFrameConditionConfig = field(
        default_factory=UNetFrameConditionConfig
    )
    vae_config: TransparentVAEConfig = field(default_factory=TransparentVAEConfig)
    text_config: SDXLTextEncoderConfig = field(default_factory=SDXLTextEncoderConfig)
    diffusion_config: DiffusionConfig = field(default_factory=DiffusionConfig)
    dtype: str = "bfloat16"

    _unet: UNetFrameConditionModel | None = field(default=None, init=False, repr=False)
    _vae: TransparentVAE | None = field(default=None, init=False, repr=False)
    _text_encoder: Any = field(default=None, init=False, repr=False)
    _text_encoder_2: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _tokenizer_2: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Load UNet + VAE + text encoders from disk.

        맥북 도착 후 채울 순서:
            1. `src.stage1_layerdiff.weights.convert_to_mlx` 또는
               이미 변환된 .npz 파일을 `mlx.core.load`로 로드
            2. 각 모듈 생성 후 `module.update(weights)` (MLX nn.Module API)
            3. dtype 적용 (`module.apply(lambda m: m.to(dtype))`)
        """
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "MLX is required for Stage 1 inference. "
                "Install on Apple Silicon with: pip install mlx"
            ) from e

        self._unet = UNetFrameConditionModel(self.unet_config)
        self._vae = TransparentVAE(self.vae_config)
        # TODO(stage1): load text encoders (CLIP-L + OpenCLIP bigG) from
        # weights_path / "text_encoder" and "text_encoder_2"

        raise NotImplementedError(
            "LayerDiffuse MLX weight loading pending — "
            "see wiki/see-through-porting.md Step 4"
        )

    def sample(
        self,
        prompt: str | list[str],
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> MLXArray:
        """Run the diffusion sampling loop to produce latents.

        High-level pseudo-code::

            # 1. Tokenise + encode text
            text_embeds, pooled = self._encode_text(prompt)
            neg_embeds, neg_pooled = self._encode_text(negative_prompt)

            # 2. Prepare latents (B, C, num_frames, H/8, W/8)
            latents = mx.random.normal(shape=..., key=rng)
            latents *= sampler.sigmas[0]

            # 3. DPM++ 2M SDE loop
            sampler = DPMSolverMultistep(self.diffusion_config, num_inference_steps)
            state = sampler.initial_state()
            for i in range(num_inference_steps):
                sigma, sigma_next = sampler.sigmas[i], sampler.sigmas[i+1]
                # CFG: duplicate batch, run both branches
                eps_cond, eps_uncond = self._unet(
                    mx.concat([latents, latents]),
                    sigma_to_timestep(sigma),
                    mx.concat([text_embeds, neg_embeds]),
                    add_text_embeds=mx.concat([pooled, neg_pooled]),
                    add_time_ids=...,
                ).split(2)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                latents = sampler.step(eps, latents, sigma, sigma_next, state)

            return latents
        """
        if self._unet is None:
            self.load()
        raise NotImplementedError(
            "LayerDiffuse sampling pending — see wiki/see-through-porting.md"
        )

    def decode_to_rgba(self, latents: MLXArray) -> np.ndarray:
        """Decode latents to RGBA numpy array `(num_frames, H, W, 4)`.

        Args:
            latents: `(1, C, num_frames, H, W)` MLX array.

        Returns:
            uint8 numpy array `(num_frames, H, W, 4)`.
        """
        if self._vae is None:
            raise RuntimeError("VAE not loaded. Call load() first.")
        raise NotImplementedError(
            "TransparentVAE decode → numpy pending — see wiki/see-through-porting.md"
        )

    def _build_sampler(self, num_inference_steps: int) -> DPMSolverMultistep:
        """Build a fresh sampler (can be called per-request)."""
        return DPMSolverMultistep(
            config=self.diffusion_config,
            num_inference_steps=num_inference_steps,
        )
