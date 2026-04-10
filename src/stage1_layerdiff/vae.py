"""TransparentVAE — LayerDiffuse의 알파 채널 VAE.

표준 SD VAE(`diffusers.models.autoencoder_kl.AutoencoderKL`)에
RGBA 출력을 추가한 형태. See-Through의 `common/modules/layerdiffuse/vae.py`가
원본.

인코더는 RGB 3채널 입력 → latent(4ch),
디코더는 latent(4ch) → RGBA 4채널 (또는 3채널+알파 head 분리).

참조:
- ml-explore/mlx-examples/.../vae.py::Autoencoder (베이스)
- diffusers.AutoencoderKL
"""

from __future__ import annotations

from typing import Any

from src.stage1_layerdiff.configs import TransparentVAEConfig

MLXArray = Any  # see mlx_ops/embeddings.py for rationale


def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError as e:
        raise RuntimeError("MLX is required. pip install mlx on Apple Silicon.") from e


class TransparentVAE:
    """LayerDiffuse 알파 VAE 래퍼.

    원본 구조:
        class TransparentVAE(AutoencoderKL):
            def __init__(self, ...):
                super().__init__(...)
                # 표준 AutoencoderKL은 out_channels=3
                # LayerDiffuse는 decoder 마지막에 알파 head 추가
                self.alpha_head = nn.Conv2d(..., 1, ...)
    """

    def __init__(self, config: TransparentVAEConfig) -> None:
        _require_mlx()
        self.config = config
        self.scaling_factor = config.scaling_factor

        # TODO(stage1): construct encoder/decoder via mlx_ops.ResnetBlock2D chain
        # Minimal shape:
        #   self.encoder = Encoder(in_channels=3, block_out_channels, layers_per_block, ...)
        #   self.decoder = Decoder(in_channels=latent_channels_in, out_channels=3, ...)
        #   self.alpha_head = nn.Conv2d(decoder_last_channels, 1, kernel_size=3, padding=1)
        #   self.quant_proj = nn.Linear(2*latent, 2*latent)
        #   self.post_quant_proj = nn.Linear(latent, latent)
        self.encoder = None
        self.decoder = None
        self.alpha_head = None
        self.quant_proj = None
        self.post_quant_proj = None

    def encode(self, x: MLXArray) -> tuple[MLXArray, MLXArray]:
        """RGB (B, 3, H, W) → (mean, logvar) in latent space.

        Returns mean and log-variance of the posterior, each
        `(B, latent_channels_in, H/8, W/8)`.
        """
        raise NotImplementedError(
            "TransparentVAE.encode pending — see wiki/see-through-porting.md"
        )

    def decode(self, z: MLXArray) -> MLXArray:
        """Latent (B, latent_channels_in, H, W) → RGBA (B, 4, H*8, W*8)."""
        raise NotImplementedError(
            "TransparentVAE.decode pending — see wiki/see-through-porting.md"
        )

    def sample_from_posterior(
        self,
        mean: MLXArray,
        logvar: MLXArray,
        noise: MLXArray | None = None,
    ) -> MLXArray:
        """Reparameterization trick: z = μ + σ · ε."""
        raise NotImplementedError(
            "TransparentVAE.sample_from_posterior pending — "
            "see wiki/see-through-porting.md"
        )
