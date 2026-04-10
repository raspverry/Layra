"""Stage 1 모델 아키텍처 dataclass configs.

MLX SD 예제(ml-explore/mlx-examples/stable_diffusion/config.py)의
`UNetConfig` / `AutoencoderConfig` / `CLIPTextModelConfig` / `DiffusionConfig`를
SDXL + LayerDiffuse(frame-conditioned) 구조로 확장한다.

**주의**: 여기 적힌 숫자들은 MLX SD 예제와 SDXL 표준을 섞은 초기값이다.
맥북 도착 후 실제 HF 허브의 `seethroughv0.0.2_layerdiff3d/unet/config.json`을
로드해서 **해당 값으로 덮어쓰는 것**이 마지막 단계. 지금은 포팅 시
인터페이스가 확정되어 있게 하는 것이 목적.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ============================================================
# VAE
# ============================================================


@dataclass(slots=True, frozen=True)
class TransparentVAEConfig:
    """LayerDiffuse의 TransparentVAE 설정.

    표준 SD VAE에 알파 채널 출력을 추가한 형태.
    `out_channels=4`로 RGBA 출력.
    """

    in_channels: int = 3
    out_channels: int = 4  # RGBA — LayerDiffuse 특유
    latent_channels_out: int = 8  # mean + logvar
    latent_channels_in: int = 4
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.13025  # SDXL VAE


# ============================================================
# Text encoder (SDXL = OpenCLIP-bigG + CLIP-L)
# ============================================================


@dataclass(slots=True, frozen=True)
class CLIPTextConfig:
    """단일 CLIP text encoder 설정."""

    num_layers: int = 32
    model_dims: int = 1280
    num_heads: int = 20
    max_length: int = 77
    vocab_size: int = 49408
    projection_dim: int | None = 1280
    hidden_act: Literal["gelu", "quick_gelu", "gelu_approx"] = "gelu"


@dataclass(slots=True, frozen=True)
class SDXLTextEncoderConfig:
    """SDXL = two text encoders (CLIP-L + OpenCLIP-bigG)."""

    # CLIP-L (768 dims)
    clip_l: CLIPTextConfig = field(
        default_factory=lambda: CLIPTextConfig(
            num_layers=12,
            model_dims=768,
            num_heads=12,
            projection_dim=None,
            hidden_act="quick_gelu",
        )
    )
    # OpenCLIP-bigG (1280 dims)
    clip_g: CLIPTextConfig = field(
        default_factory=lambda: CLIPTextConfig(
            num_layers=32,
            model_dims=1280,
            num_heads=20,
            projection_dim=1280,
            hidden_act="gelu",
        )
    )
    # SDXL은 두 인코더의 hidden을 concat → 768 + 1280 = 2048
    concat_dim: int = 2048


# ============================================================
# UNet — LayerDiffuse frame-conditioned SDXL UNet
# ============================================================


@dataclass(slots=True, frozen=True)
class UNetFrameConditionConfig:
    """UNetFrameConditionModel 아키텍처 설정.

    표준 SDXL UNet + frame(layer) 차원 추가. num_frames는
    See-Through의 23개 레이어 구조를 의미한다.
    """

    # --- Frame / layer 차원 ---
    num_frames: int = 23
    """See-Through 표준 레이어 수."""

    # --- 기본 SDXL UNet ---
    in_channels: int = 4
    out_channels: int = 4
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3

    # SDXL 표준 3-stage 구조
    block_out_channels: tuple[int, ...] = (320, 640, 1280)
    layers_per_block: tuple[int, ...] = (2, 2, 2)
    mid_block_layers: int = 2

    # Attention per block (SDXL: 마지막 두 스테이지에만 transformer)
    transformer_layers_per_block: tuple[int, ...] = (0, 2, 10)
    num_attention_heads: tuple[int, ...] = (5, 10, 20)

    # SDXL cross-attention dim (CLIP-L + bigG concat)
    cross_attention_dim: int = 2048

    # Norm
    norm_num_groups: int = 32

    # SDXL 추가 conditioning
    addition_embed_type: Literal["text_time", "none"] = "text_time"
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int = 2816  # 1280 + 1536

    # --- LayerDiffuse 특유: cross-frame attention ---
    use_cross_frame_attention: bool = True
    """True면 TemporalBasicTransformerBlock + CrossFrameTransformerBlock
    을 사용해 레이어들 간 일관성을 보장한다."""

    cross_frame_attention_layers: tuple[str, ...] = ("self", "cross")
    """어느 attention에 frame 차원 공유를 적용할지."""


# ============================================================
# Marigold (fine-tuned depth diffusion)
# ============================================================


@dataclass(slots=True, frozen=True)
class MarigoldConfig:
    """prs-eth/marigold 기반. SD 1.5 백본 + depth 출력.

    See-Through fine-tune은 레이어별 pseudo-depth를 추정해
    drawing order 결정에 사용된다.
    """

    in_channels: int = 8  # latent(4) + image latent(4) concat
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    cross_attention_dim: int = 768  # SD 1.5
    num_attention_heads: tuple[int, ...] = (5, 10, 20, 20)
    norm_num_groups: int = 32
    resolution: int = 768
    num_inference_steps: int = 10  # Marigold는 짧은 step으로도 OK
    ensemble_size: int = 5  # inference ensemble


# ============================================================
# Diffusion (k-diffusion DPM++ 2M SDE)
# ============================================================


@dataclass(slots=True, frozen=True)
class DiffusionConfig:
    """See-Through의 KDiffusionStableDiffusionXLPipeline 설정.

    원본은 `DPMPP_2M_SDE` + `final_sigmas_type="zero"` + Karras sigmas.
    """

    beta_schedule: Literal["linear", "scaled_linear"] = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_steps: int = 1000
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon"

    # k-diffusion sampler
    sampler: Literal[
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_karras", "euler", "euler_a"
    ] = "dpmpp_2m_sde"
    use_karras_sigmas: bool = True
    final_sigmas_type: Literal["zero", "sigma_min"] = "zero"
