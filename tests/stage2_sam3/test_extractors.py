"""Tests for NeckExtractor and MouthExtractor.

These tests inject a `Sam3TextExtractor` backend whose processor is a
mock, so no SAM3 weights or segment_anything_3 package are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.config import Stage2Config
from src.stage2_sam3 import MouthExtractor, NeckExtractor, Sam3TextExtractor


@dataclass
class MockSam3Processor:
    returned_masks: np.ndarray
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def set_image(self, image: np.ndarray) -> dict[str, Any]:
        self.calls.append(("set_image", image.shape))
        return {}

    def reset_all_prompts(self, state: dict[str, Any]) -> None:
        pass

    def set_text_prompt(self, state: dict[str, Any], prompt: str) -> dict[str, Any]:
        self.calls.append(("set_text_prompt", prompt))
        state["masks"] = self.returned_masks
        return state


def _make_backend(returned: np.ndarray) -> tuple[Sam3TextExtractor, MockSam3Processor]:
    cfg = Stage2Config()
    backend = Sam3TextExtractor(config=cfg)
    mock = MockSam3Processor(returned_masks=returned)
    backend.inject_processor(mock)
    return backend, mock


class TestNeckExtractor:
    def test_extract_uses_neck_prompt(self) -> None:
        returned = np.ones((8, 8), dtype=np.uint8)
        backend, mock = _make_backend(returned)

        extractor = NeckExtractor(config=Stage2Config())
        extractor.inject_backend(backend)

        out = extractor.extract(np.zeros((8, 8, 3), dtype=np.uint8))
        assert out.dtype == np.uint8
        assert mock.calls[-1] == ("set_text_prompt", "neck")

    def test_extract_returns_postprocessed_shape(self) -> None:
        returned = np.zeros((16, 16), dtype=np.uint8)
        returned[6:10, 6:10] = 1
        backend, _ = _make_backend(returned)

        extractor = NeckExtractor(config=Stage2Config())
        extractor.inject_backend(backend)
        out = extractor.extract(np.zeros((16, 16, 3), dtype=np.uint8))
        assert out.shape == (16, 16)
        # dilate + blur = strictly more non-zero pixels than input
        assert (out > 0).sum() >= (returned > 0).sum()


class TestMouthExtractor:
    def test_extract_uses_mouth_prompt(self) -> None:
        returned = np.ones((8, 8), dtype=np.uint8)
        backend, mock = _make_backend(returned)

        extractor = MouthExtractor(config=Stage2Config())
        extractor.inject_backend(backend)
        extractor.extract(np.zeros((8, 8, 3), dtype=np.uint8))
        assert mock.calls[-1] == ("set_text_prompt", "mouth")

    def test_extract_pair_returns_two_masks(self) -> None:
        returned = np.ones((4, 4), dtype=np.uint8)
        backend, mock = _make_backend(returned)

        extractor = MouthExtractor(config=Stage2Config())
        extractor.inject_backend(backend)

        closed = np.zeros((4, 4, 3), dtype=np.uint8)
        opened = np.ones((4, 4, 3), dtype=np.uint8) * 255
        m_closed, m_opened = extractor.extract_pair(closed, opened)

        assert m_closed.shape == (4, 4)
        assert m_opened.shape == (4, 4)
        # Both invocations should have sent the "mouth" prompt
        mouth_calls = [c for c in mock.calls if c[0] == "set_text_prompt"]
        assert len(mouth_calls) == 2
        assert all(c == ("set_text_prompt", "mouth") for c in mouth_calls)
