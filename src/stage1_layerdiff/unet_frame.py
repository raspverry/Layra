"""UNetFrameConditionModel — LayerDiffuse의 SDXL+frame UNet.

MLX SD `UNetModel`을 베이스로, 레이어 차원(num_frames=23)을 처리하도록
확장한 MLX 버전 스켈레톤.

원본:
    shitagaki-lab/see-through/common/modules/layerdiffuse/layerdiff3d.py
        class UNetFrameConditionModel(ModelMixin, ConfigMixin):
            ...

이 파일은 맥북 도착 후 채워진다. 지금은:
  - class 구조, 속성, 생성자 시그니처만 정의
  - __call__은 NotImplementedError
  - linux 환경에서 import/mypy/ruff 전부 통과해야 함
"""

from __future__ import annotations

from typing import Any

from src.stage1_layerdiff.configs import UNetFrameConditionConfig
from src.stage1_layerdiff.mlx_ops import (
    ResnetBlock2D,
    SDXLAdditionEmbedding,
    TimestepEmbedding,
    Transformer3DModel,
    UNetBlock2D,
)

MLXArray = Any  # see mlx_ops/embeddings.py for rationale


def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MLX is required for Stage 1. Install on Apple Silicon: pip install mlx"
        ) from e


class UNetFrameConditionModel:
    """See-Through LayerDiffuse SDXL UNet (frame-conditioned).

    Forward shape convention:
        x:            (B, C, num_frames, H, W)        — frame-axis latent
        timestep:     (B,)                            — int32 diffusion step
        encoder_x:    (B, seq_len, cross_attn_dim)    — text conditioning
        add_text_embeds: (B, projection_dim)          — SDXL pooled text
        add_time_ids:    (B, 6)                       — SDXL size embedding

    Output:
        eps_pred:     (B, C, num_frames, H, W)        — noise prediction
    """

    def __init__(self, config: UNetFrameConditionConfig) -> None:
        _require_mlx()
        self.config = config

        # Derived dims
        self.time_embed_dim = config.block_out_channels[0] * 4
        self.num_frames = config.num_frames

        # TODO(stage1): construct all submodules with mlx.nn
        #
        # Top-level layout mirrors MLX SD UNetModel + SDXL additions:
        #
        # self.conv_in = nn.Conv2d(
        #     config.in_channels,
        #     config.block_out_channels[0],
        #     kernel_size=config.conv_in_kernel,
        #     padding=config.conv_in_kernel // 2,
        # )
        # self.time_proj = partial(timestep_embedding, dim=config.block_out_channels[0])
        # self.time_embedding = TimestepEmbedding(
        #     config.block_out_channels[0], self.time_embed_dim
        # )
        # self.add_embedding = SDXLAdditionEmbedding(
        #     config.addition_time_embed_dim,
        #     config.projection_class_embeddings_input_dim,
        #     self.time_embed_dim,
        # )
        # self.down_blocks = [
        #     UNetBlock2D(
        #         in_channels=config.block_out_channels[max(0, i - 1)],
        #         out_channels=config.block_out_channels[i],
        #         temb_channels=self.time_embed_dim,
        #         num_layers=config.layers_per_block[i],
        #         transformer_layers_per_block=config.transformer_layers_per_block[i],
        #         num_attention_heads=config.num_attention_heads[i],
        #         cross_attention_dim=config.cross_attention_dim,
        #         resnet_groups=config.norm_num_groups,
        #         add_downsample=(i < len(config.block_out_channels) - 1),
        #         add_upsample=False,
        #         add_cross_attention=(
        #             config.transformer_layers_per_block[i] > 0
        #         ),
        #         num_frames=config.num_frames,
        #     )
        #     for i in range(len(config.block_out_channels))
        # ]
        # self.mid_blocks = [
        #     ResnetBlock2D(
        #         config.block_out_channels[-1],
        #         temb_channels=self.time_embed_dim,
        #         groups=config.norm_num_groups,
        #     ),
        #     Transformer3DModel(
        #         in_channels=config.block_out_channels[-1],
        #         num_layers=config.mid_block_layers,
        #         num_attention_heads=config.num_attention_heads[-1],
        #         attention_head_dim=config.block_out_channels[-1]
        #             // config.num_attention_heads[-1],
        #         cross_attention_dim=config.cross_attention_dim,
        #         num_frames=config.num_frames,
        #         norm_num_groups=config.norm_num_groups,
        #         use_cross_frame_attention=config.use_cross_frame_attention,
        #     ),
        #     ResnetBlock2D(
        #         config.block_out_channels[-1],
        #         temb_channels=self.time_embed_dim,
        #         groups=config.norm_num_groups,
        #     ),
        # ]
        # self.up_blocks = [... reversed ...]
        # self.conv_norm_out = nn.GroupNorm(config.norm_num_groups, config.block_out_channels[0])
        # self.conv_out = nn.Conv2d(
        #     config.block_out_channels[0], config.out_channels, kernel_size=3, padding=1
        # )

        # Placeholders so typechecker sees the attributes
        self.conv_in = None
        self.time_embedding: TimestepEmbedding | None = None
        self.add_embedding: SDXLAdditionEmbedding | None = None
        self.down_blocks: list[UNetBlock2D] = []
        self.mid_blocks: list[ResnetBlock2D | Transformer3DModel] = []
        self.up_blocks: list[UNetBlock2D] = []
        self.conv_norm_out = None
        self.conv_out = None

    def __call__(
        self,
        x: MLXArray,
        timestep: MLXArray,
        encoder_x: MLXArray,
        add_text_embeds: MLXArray | None = None,
        add_time_ids: MLXArray | None = None,
        attn_mask: MLXArray | None = None,
        encoder_attn_mask: MLXArray | None = None,
    ) -> MLXArray:
        """Forward pass.

        Pseudo-code (from MLX SD + SDXL + transformer3d.py):

            # 1. Time embedding
            t_emb = timestep_embedding(timestep, dim=block_out_channels[0])
            t_emb = self.time_embedding(t_emb)

            # 2. SDXL addition embedding (pooled CLIP + size embeds)
            if self.config.addition_embed_type == "text_time":
                aug = self.add_embedding(add_text_embeds, add_time_ids)
                t_emb = t_emb + aug

            # 3. conv_in
            h = self.conv_in(x)  # frame axis passes through

            # 4. Down blocks — collect residuals
            residuals = [h]
            for block in self.down_blocks:
                h, block_residuals = block(h, t_emb, encoder_x,
                                           attn_mask, encoder_attn_mask)
                residuals.extend(block_residuals)

            # 5. Mid
            h = self.mid_blocks[0](h, t_emb)
            h = self.mid_blocks[1](h, encoder_x, attn_mask, encoder_attn_mask)
            h = self.mid_blocks[2](h, t_emb)

            # 6. Up blocks — consume residuals
            for block in self.up_blocks:
                h = block(h, t_emb, encoder_x,
                          attn_mask, encoder_attn_mask,
                          residual_hidden_states=residuals)

            # 7. Output
            h = self.conv_norm_out(h)
            h = nn.silu(h)
            h = self.conv_out(h)
            return h
        """
        raise NotImplementedError(
            "UNetFrameConditionModel forward pending — "
            "see wiki/see-through-porting.md Step 7"
        )
