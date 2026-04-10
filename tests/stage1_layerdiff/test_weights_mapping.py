"""Tests for weight key remapping (PyTorch → MLX).

No PyTorch or MLX needed — purely string/dict manipulation.
"""

from __future__ import annotations

from src.stage1_layerdiff.weights import (
    UNET_KEY_MAP,
    VAE_KEY_MAP,
    convert_state_dict,
    rewrite_key,
)


class TestRewriteKey:
    def test_unet_conv_in(self) -> None:
        out = rewrite_key("conv_in.weight", UNET_KEY_MAP)
        assert out == "conv_in.weight"

    def test_unet_mid_block_resnet_0_to_mid_blocks_0(self) -> None:
        out = rewrite_key("mid_block.resnets.0.conv1.weight", UNET_KEY_MAP)
        assert out == "mid_blocks.0.conv1.weight"

    def test_unet_mid_block_attention_to_mid_blocks_1(self) -> None:
        out = rewrite_key("mid_block.attentions.0.proj_in.weight", UNET_KEY_MAP)
        assert out == "mid_blocks.1.proj_in.weight"

    def test_unet_mid_block_resnet_1_to_mid_blocks_2(self) -> None:
        out = rewrite_key("mid_block.resnets.1.conv2.bias", UNET_KEY_MAP)
        assert out == "mid_blocks.2.conv2.bias"

    def test_unet_unknown_prefix_drops(self) -> None:
        out = rewrite_key("completely_unknown.weight", UNET_KEY_MAP)
        assert out is None

    def test_vae_quant_conv_rename(self) -> None:
        out = rewrite_key("quant_conv.weight", VAE_KEY_MAP)
        assert out == "quant_proj.weight"

    def test_vae_post_quant_conv_rename(self) -> None:
        out = rewrite_key("post_quant_conv.bias", VAE_KEY_MAP)
        assert out == "post_quant_proj.bias"


class TestConvertStateDict:
    def test_unet_basic_mapping(self) -> None:
        sd = {
            "conv_in.weight": "W1",
            "mid_block.resnets.0.conv1.weight": "W2",
            "mid_block.attentions.0.proj_in.weight": "W3",
            "unknown_key": "W_drop",
        }
        out = convert_state_dict(sd, "unet")
        assert out == {
            "conv_in.weight": "W1",
            "mid_blocks.0.conv1.weight": "W2",
            "mid_blocks.1.proj_in.weight": "W3",
        }

    def test_vae_basic_mapping(self) -> None:
        sd = {
            "encoder.conv_in.weight": "W1",
            "quant_conv.weight": "W2",
            "post_quant_conv.bias": "W3",
            "alpha_head.weight": "W4",
        }
        out = convert_state_dict(sd, "vae")
        assert out == {
            "encoder.conv_in.weight": "W1",
            "quant_proj.weight": "W2",
            "post_quant_proj.bias": "W3",
            "alpha_head.weight": "W4",
        }

    def test_dropped_callback(self) -> None:
        dropped: list[str] = []
        sd = {"conv_in.weight": "W1", "junk": "W2"}
        convert_state_dict(sd, "unet", dropped_callback=dropped.append)
        assert dropped == ["junk"]
