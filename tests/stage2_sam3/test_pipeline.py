"""End-to-end tests for Stage2Pipeline.

Writes a synthetic per-layer PSD + a synthetic RGB image, injects mock
extractors, and verifies:
  - PSD → parsed LayerSet → body/hair/hair_back PNG files
  - Optional mouth_pair path (when open_mouth_image is provided)
  - All outputs are valid uint8 RGBA shapes
  - Returned Stage2Output paths match disk state

No SAM3 weights or segment_anything_3 package are needed — the extractor
backends are mocked via Sam3TextExtractor.inject_processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import Stage2Config
from src.common.image_io import load_rgba, save_rgb
from src.common.psd_io import write_psd
from src.common.types import LayerName, LayerSet
from src.stage2_sam3 import (
    MouthExtractor,
    NeckExtractor,
    Sam3TextExtractor,
    Stage2Pipeline,
)

# ============================================================
# Mock SAM3 processor
# ============================================================


@dataclass
class _MockProcessor:
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


def _make_extractor(returned: np.ndarray) -> Sam3TextExtractor:
    backend = Sam3TextExtractor(config=Stage2Config())
    backend.inject_processor(_MockProcessor(returned_masks=returned))
    return backend


# ============================================================
# Fixtures: synthetic input image + PSD
# ============================================================


def _make_layerset(h: int, w: int) -> LayerSet:
    """Build a synthetic LayerSet with body + hair_front + hair_back."""
    ls = LayerSet()

    body = np.zeros((h, w, 4), dtype=np.uint8)
    body[:, :, 0] = 200  # R
    body[:, :, 3] = 255
    ls.layers[LayerName.BODY] = body

    hair_front = np.zeros((h, w, 4), dtype=np.uint8)
    hair_front[: h // 3, :, 2] = 255  # 상단 파랑
    hair_front[: h // 3, :, 3] = 255
    ls.layers[LayerName.HAIR_FRONT] = hair_front

    hair_back = np.zeros((h, w, 4), dtype=np.uint8)
    hair_back[:, : w // 4, 1] = 180  # 왼쪽 초록
    hair_back[:, : w // 4, 3] = 255
    ls.layers[LayerName.HAIR_BACK] = hair_back

    ls.drawing_order = [
        LayerName.HAIR_BACK,
        LayerName.BODY,
        LayerName.HAIR_FRONT,
    ]
    return ls


def _write_synthetic_inputs(
    tmp_path: Path, h: int = 48, w: int = 64
) -> tuple[Path, Path]:
    """Return (image_path, psd_path) of a synthetic input pair."""
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        image[y, :, 0] = (y * 255) // h
        image[y, :, 1] = 100
        image[y, :, 2] = 255 - (y * 255) // h

    image_path = tmp_path / "input.png"
    save_rgb(image, image_path)

    psd_path = tmp_path / "input.psd"
    write_psd(_make_layerset(h, w), psd_path)

    return image_path, psd_path


# ============================================================
# Tests
# ============================================================


class TestStage2PipelineEndToEnd:
    def test_writes_body_hair_outputs(self, tmp_path: Path) -> None:
        h, w = 48, 64
        image_path, psd_path = _write_synthetic_inputs(tmp_path, h, w)

        # Neck mask: 전체 이미지의 절반 아래쪽 (목 영역 시뮬레이션)
        neck_mask_mock = np.zeros((h, w), dtype=np.uint8)
        neck_mask_mock[h // 2 :, :] = 1

        cfg = Stage2Config()
        neck = NeckExtractor(config=cfg)
        neck.inject_backend(_make_extractor(neck_mask_mock))

        out_dir = tmp_path / "out"
        pipeline = Stage2Pipeline(config=cfg, neck_extractor=neck)
        result = pipeline(
            original_image=image_path,
            psd_path=psd_path,
            output_dir=out_dir,
        )

        # 필수 파일들 생성됐는지
        assert result.body_png.exists()
        assert result.hair_front_png.exists()
        assert result.hair_back_png.exists()
        assert result.mouth_closed_png is None
        assert result.mouth_open_png is None
        assert result.elapsed_seconds >= 0.0

        # 출력 shape 검증
        body = load_rgba(result.body_png)
        assert body.shape == (h, w, 4)
        assert body.dtype == np.uint8

        hair_front = load_rgba(result.hair_front_png)
        assert hair_front.shape == (h, w, 4)
        hair_back = load_rgba(result.hair_back_png)
        assert hair_back.shape == (h, w, 4)

    def test_mouth_pair_when_open_image_provided(self, tmp_path: Path) -> None:
        h, w = 48, 64
        image_path, psd_path = _write_synthetic_inputs(tmp_path, h, w)

        # Second image for "mouth open" state
        open_image = np.ones((h, w, 3), dtype=np.uint8) * 128
        open_image_path = tmp_path / "open_mouth.png"
        save_rgb(open_image, open_image_path)

        # Both extractors get their own mock backend
        neck_mask = np.ones((h, w), dtype=np.uint8)
        mouth_mask = np.zeros((h, w), dtype=np.uint8)
        mouth_mask[h // 4 : h // 2, w // 4 : 3 * w // 4] = 1

        cfg = Stage2Config()
        neck = NeckExtractor(config=cfg)
        neck.inject_backend(_make_extractor(neck_mask))
        mouth = MouthExtractor(config=cfg)
        mouth.inject_backend(_make_extractor(mouth_mask))

        out_dir = tmp_path / "out"
        pipeline = Stage2Pipeline(
            config=cfg, neck_extractor=neck, mouth_extractor=mouth
        )
        result = pipeline(
            original_image=image_path,
            psd_path=psd_path,
            output_dir=out_dir,
            open_mouth_image=open_image_path,
        )

        assert result.mouth_closed_png is not None
        assert result.mouth_closed_png.exists()
        assert result.mouth_open_png is not None
        assert result.mouth_open_png.exists()

        closed = load_rgba(result.mouth_closed_png)
        opened = load_rgba(result.mouth_open_png)
        assert closed.shape == (h, w, 4)
        assert opened.shape == (h, w, 4)
        # 두 이미지의 RGB 베이스가 다르면 결과 RGB도 다르다
        assert not np.array_equal(closed[..., :3], opened[..., :3])

    def test_pipeline_without_hair_back(self, tmp_path: Path) -> None:
        """hair_back 레이어 없는 PSD에서도 실패하지 않아야 한다."""
        h, w = 32, 32
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image_path = tmp_path / "input.png"
        save_rgb(image, image_path)

        ls = LayerSet()
        body = np.zeros((h, w, 4), dtype=np.uint8)
        body[..., 3] = 255
        ls.layers[LayerName.BODY] = body
        ls.drawing_order = [LayerName.BODY]
        psd_path = tmp_path / "input.psd"
        write_psd(ls, psd_path)

        cfg = Stage2Config()
        neck = NeckExtractor(config=cfg)
        neck.inject_backend(_make_extractor(np.ones((h, w), dtype=np.uint8)))

        out_dir = tmp_path / "out"
        pipeline = Stage2Pipeline(config=cfg, neck_extractor=neck)
        result = pipeline(
            original_image=image_path,
            psd_path=psd_path,
            output_dir=out_dir,
        )
        assert result.body_png.exists()
        # hair_back_png는 파일이 생성되지 않은 경로로 반환됨
        assert not result.hair_back_png.exists()
