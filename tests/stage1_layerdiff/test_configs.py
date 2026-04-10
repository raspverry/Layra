"""src/stage1_layerdiff/configs.py 테스트."""

from __future__ import annotations

from src.stage1_layerdiff.configs import (
    DiffusionConfig,
    MarigoldConfig,
    SDXLTextEncoderConfig,
    TransparentVAEConfig,
    UNetFrameConditionConfig,
)


class TestUNetFrameConditionConfig:
    def test_defaults_match_sdxl_layout(self) -> None:
        cfg = UNetFrameConditionConfig()
        assert cfg.num_frames == 23  # See-Through standard
        assert cfg.in_channels == 4
        assert cfg.out_channels == 4
        assert cfg.cross_attention_dim == 2048  # SDXL
        assert cfg.block_out_channels == (320, 640, 1280)
        # SDXL-style: 첫 스테이지는 transformer 없음
        assert cfg.transformer_layers_per_block[0] == 0
        assert cfg.num_attention_heads == (5, 10, 20)

    def test_addition_embed_type(self) -> None:
        cfg = UNetFrameConditionConfig()
        assert cfg.addition_embed_type == "text_time"
        assert cfg.projection_class_embeddings_input_dim > 0

    def test_num_frames_matches_layer_enum(self) -> None:
        """num_frames는 23으로 LayerName enum 카운트와 일치해야 한다."""
        from src.common.types import LayerName

        cfg = UNetFrameConditionConfig()
        assert cfg.num_frames == len(LayerName)

    def test_is_frozen(self) -> None:
        """frozen=True 보장 — 실수로 값이 바뀌면 안 됨."""
        import dataclasses

        cfg = UNetFrameConditionConfig()
        try:
            cfg.num_frames = 24  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("UNetFrameConditionConfig should be frozen")


class TestTransparentVAEConfig:
    def test_rgba_output(self) -> None:
        cfg = TransparentVAEConfig()
        assert cfg.in_channels == 3
        assert cfg.out_channels == 4  # RGBA (LayerDiffuse)

    def test_sdxl_scaling_factor(self) -> None:
        cfg = TransparentVAEConfig()
        # SDXL uses 0.13025 (not 0.18215 of SD 1.x)
        assert abs(cfg.scaling_factor - 0.13025) < 1e-6


class TestSDXLTextEncoderConfig:
    def test_two_encoders(self) -> None:
        cfg = SDXLTextEncoderConfig()
        assert cfg.clip_l.model_dims == 768
        assert cfg.clip_g.model_dims == 1280
        # concat dim = 768 + 1280 = 2048
        assert cfg.concat_dim == 2048

    def test_clip_l_quick_gelu(self) -> None:
        """CLIP-L uses quick_gelu (original CLIP), bigG uses gelu."""
        cfg = SDXLTextEncoderConfig()
        assert cfg.clip_l.hidden_act == "quick_gelu"
        assert cfg.clip_g.hidden_act == "gelu"


class TestDiffusionConfig:
    def test_scaled_linear_defaults(self) -> None:
        cfg = DiffusionConfig()
        assert cfg.beta_schedule == "scaled_linear"
        assert cfg.beta_start == 0.00085
        assert cfg.beta_end == 0.012
        assert cfg.num_train_steps == 1000

    def test_dpmpp_2m_sde_default(self) -> None:
        cfg = DiffusionConfig()
        assert cfg.sampler == "dpmpp_2m_sde"
        assert cfg.use_karras_sigmas is True
        assert cfg.final_sigmas_type == "zero"


class TestMarigoldConfig:
    def test_sd15_backbone(self) -> None:
        cfg = MarigoldConfig()
        # Marigold는 SD 1.5 기반 → cross_attention_dim 768
        assert cfg.cross_attention_dim == 768
        assert cfg.block_out_channels == (320, 640, 1280, 1280)  # SD 1.5 layout

    def test_ensemble_defaults(self) -> None:
        cfg = MarigoldConfig()
        assert cfg.ensemble_size >= 1
        assert cfg.num_inference_steps >= 1
