"""Tests for src/stage2_sam3/sam3_backend.py.

These tests do not require SAM3 weights or the segment_anything_3 package.
They use a minimal mock processor that implements the Sam3ProcessorLike
protocol and verify both the extraction pipeline and the mask
postprocessing math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from src.common.config import Stage2Config
from src.stage2_sam3.sam3_backend import (
    Sam3TextExtractor,
    combine_masks,
    extract_with_processor,
    postprocess_mask,
)

# ============================================================
# Mock processor — Sam3ProcessorLike protocol
# ============================================================


@dataclass
class MockSam3Processor:
    """Minimal Sam3Processor stand-in for tests."""

    returned_masks: np.ndarray
    """Masks to be returned by set_text_prompt, shape (N, H, W) or (H, W)."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    """(method_name, args) log for behavioural assertions."""

    last_state_key: int = 0

    def set_image(self, image: np.ndarray) -> dict[str, Any]:
        self.calls.append(("set_image", image.shape))
        self.last_state_key += 1
        return {"_id": self.last_state_key, "image_shape": image.shape}

    def reset_all_prompts(self, state: dict[str, Any]) -> None:
        self.calls.append(("reset_all_prompts", state["_id"]))
        state.pop("prompt", None)

    def set_text_prompt(self, state: dict[str, Any], prompt: str) -> dict[str, Any]:
        self.calls.append(("set_text_prompt", prompt))
        state["prompt"] = prompt
        state["masks"] = self.returned_masks
        return state


# ============================================================
# combine_masks
# ============================================================


class TestCombineMasks:
    def test_single_2d_mask(self) -> None:
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        out = combine_masks(mask)
        np.testing.assert_array_equal(out, [[0, 1], [1, 0]])
        assert out.dtype == np.uint8

    def test_stacked_masks_or(self) -> None:
        stacked = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=np.uint8,
        )
        out = combine_masks(stacked)
        np.testing.assert_array_equal(out, [[1, 1], [1, 0]])

    def test_float_mask_thresholded(self) -> None:
        stacked = np.array([[[0.1, 0.0], [0.8, 0.0]]], dtype=np.float32)
        out = combine_masks(stacked)
        # 0.1 > 0 → 1
        np.testing.assert_array_equal(out, [[1, 0], [1, 0]])

    def test_invalid_ndim(self) -> None:
        with pytest.raises(ValueError, match="Unexpected mask ndim"):
            combine_masks(np.zeros((1, 1, 1, 1)))


# ============================================================
# postprocess_mask
# ============================================================


class TestPostprocessMask:
    def test_dilate_and_blur_preserves_uint8(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        out = postprocess_mask(mask, dilate_iterations=2, blur_kernel=7)
        assert out.dtype == np.uint8
        assert out.shape == (32, 32)
        # Dilate + blur expands the region of non-zero pixels.
        assert (out > 0).sum() > (mask > 0).sum()

    def test_skip_dilate(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[5:8, 5:8] = 1
        out = postprocess_mask(mask, dilate_iterations=0, blur_kernel=0)
        # No dilate, no blur → exactly 255 where mask was set
        expected = mask * 255
        np.testing.assert_array_equal(out, expected)

    def test_even_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="blur_kernel must be odd"):
            postprocess_mask(
                np.zeros((8, 8), dtype=np.uint8),
                dilate_iterations=0,
                blur_kernel=4,
            )

    def test_bool_input_accepted(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:8, 4:8] = True
        out = postprocess_mask(mask, dilate_iterations=1, blur_kernel=0)
        assert out.dtype == np.uint8
        assert out.max() == 255


# ============================================================
# extract_with_processor
# ============================================================


class TestExtractWithProcessor:
    def test_basic_flow(self) -> None:
        returned = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        processor = MockSam3Processor(returned_masks=returned)
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        out = extract_with_processor(processor, image, prompt="neck")
        np.testing.assert_array_equal(out, [[1, 1], [0, 0]])

        # Verify call sequence: set_image → reset_all_prompts → set_text_prompt
        assert [c[0] for c in processor.calls] == [
            "set_image",
            "reset_all_prompts",
            "set_text_prompt",
        ]
        assert processor.calls[-1] == ("set_text_prompt", "neck")

    def test_multiple_masks_combined(self) -> None:
        stacked = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        processor = MockSam3Processor(returned_masks=stacked)
        out = extract_with_processor(
            processor, np.zeros((2, 2, 3), dtype=np.uint8), prompt="mouth"
        )
        np.testing.assert_array_equal(out, [[1, 0], [0, 1]])

    def test_pred_masks_fallback_key(self) -> None:
        """Some SAM3 builds return `pred_masks` instead of `masks`."""

        class AltProcessor:
            def set_image(self, image: np.ndarray) -> dict[str, Any]:
                return {}

            def reset_all_prompts(self, state: dict[str, Any]) -> None:
                pass

            def set_text_prompt(
                self, state: dict[str, Any], prompt: str
            ) -> dict[str, Any]:
                state["pred_masks"] = np.ones((4, 4), dtype=np.uint8)
                return state

        out = extract_with_processor(
            AltProcessor(), np.zeros((4, 4, 3), dtype=np.uint8), prompt="eye"
        )
        assert out.shape == (4, 4)
        assert (out == 1).all()

    def test_empty_state_raises(self) -> None:
        class EmptyProcessor:
            def set_image(self, image: np.ndarray) -> dict[str, Any]:
                return {}

            def reset_all_prompts(self, state: dict[str, Any]) -> None:
                pass

            def set_text_prompt(
                self, state: dict[str, Any], prompt: str
            ) -> dict[str, Any]:
                return state  # no "masks" key

        with pytest.raises(RuntimeError, match="no masks"):
            extract_with_processor(
                EmptyProcessor(),
                np.zeros((4, 4, 3), dtype=np.uint8),
                prompt="nose",
            )


# ============================================================
# Sam3TextExtractor with injected mock
# ============================================================


class TestSam3TextExtractor:
    def _make(
        self, returned: np.ndarray, **config_overrides: Any
    ) -> tuple[Sam3TextExtractor, MockSam3Processor]:
        cfg = Stage2Config(**config_overrides)
        extractor = Sam3TextExtractor(config=cfg)
        mock = MockSam3Processor(returned_masks=returned)
        extractor.inject_processor(mock)
        return extractor, mock

    def test_extract_runs_postprocess(self) -> None:
        returned = np.zeros((8, 8), dtype=np.uint8)
        returned[3:5, 3:5] = 1
        extractor, mock = self._make(returned)

        out = extractor.extract(np.zeros((8, 8, 3), dtype=np.uint8), prompt="neck")
        assert out.dtype == np.uint8
        # Dilate (2 iter) + blur expands the region
        assert (out > 0).sum() > (returned > 0).sum()
        assert mock.calls[-1] == ("set_text_prompt", "neck")

    def test_extract_named_resolves_prompt(self) -> None:
        returned = np.ones((4, 4), dtype=np.uint8)
        extractor, mock = self._make(returned)

        extractor.extract_named(np.zeros((4, 4, 3), dtype=np.uint8), "neck")
        assert mock.calls[-1] == ("set_text_prompt", "neck")

    def test_extract_named_falls_back_to_raw(self) -> None:
        """part_name이 config.text_prompts에 없으면 그대로 사용."""
        returned = np.ones((4, 4), dtype=np.uint8)
        extractor, mock = self._make(returned, text_prompts={})
        extractor.extract_named(np.zeros((4, 4, 3), dtype=np.uint8), "hair_front")
        assert mock.calls[-1] == ("set_text_prompt", "hair_front")

    def test_extract_named_custom_prompt(self) -> None:
        returned = np.ones((4, 4), dtype=np.uint8)
        extractor, mock = self._make(
            returned,
            text_prompts={"neck": "neck,throat"},
        )
        extractor.extract_named(np.zeros((4, 4, 3), dtype=np.uint8), "neck")
        assert mock.calls[-1] == ("set_text_prompt", "neck,throat")

    def test_inject_processor_skips_ensure_loaded(self) -> None:
        """Injecting a processor should bypass disk loading entirely."""
        returned = np.ones((4, 4), dtype=np.uint8)
        extractor, _ = self._make(returned)
        # SAM3 weights don't exist on Linux — calling _ensure_loaded would
        # raise FileNotFoundError. Injection should mean we never hit that.
        extractor.extract(
            np.zeros((4, 4, 3), dtype=np.uint8), prompt="neck"
        )  # should not raise
