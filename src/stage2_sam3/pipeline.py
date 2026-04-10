"""Stage 2 오케스트레이션.

PSD → LayerSet → SAM3 보정 → body.png / hair.png / hair_back.png 저장.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.common.config import Stage2Config, get_config
from src.common.image_io import load_rgb, save_rgba
from src.common.logging import get_logger
from src.common.psd_io import parse_psd
from src.common.types import Stage2Output
from src.stage2_sam3.mouth_extractor import MouthExtractor
from src.stage2_sam3.neck_extractor import NeckExtractor
from src.stage2_sam3.psd_parser import layerset_to_body_parts

logger = get_logger(__name__)


@dataclass(slots=True)
class Stage2Pipeline:
    """Stage 2 end-to-end 파이프라인."""

    config: Stage2Config

    def __call__(
        self,
        original_image: Path,
        psd_path: Path,
        output_dir: Path,
        open_mouth_image: Path | None = None,
    ) -> Stage2Output:
        """Stage 2 실행.

        Args:
            original_image: Stage 1에 넣었던 원본 PNG.
            psd_path: Stage 1이 생성한 PSD 파일.
            output_dir: 출력 디렉토리.
            open_mouth_image: (선택) 사용자가 제공한 '열린 입' 이미지.
                있으면 MouthExtractor로 mouth pair 추출.

        Returns:
            Stage2Output.
        """
        import numpy as np

        start = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Stage 2 start: psd={psd_path.name}")

        rgb = load_rgb(original_image)
        layer_set = parse_psd(psd_path)
        parts = layerset_to_body_parts(layer_set)

        neck_extractor = NeckExtractor(config=self.config)
        neck_mask = neck_extractor.extract(rgb)

        body_rgba = _apply_mask_to_rgb(rgb, neck_mask, base=parts.body)
        body_path = output_dir / "body.png"
        save_rgba(body_rgba, body_path)

        hair_front_path = output_dir / "hair.png"
        if parts.hair_front is not None:
            save_rgba(parts.hair_front, hair_front_path)

        hair_back_path = output_dir / "hair_back.png"
        if parts.hair_back is not None:
            save_rgba(parts.hair_back, hair_back_path)

        mouth_closed_path: Path | None = None
        mouth_open_path: Path | None = None
        if open_mouth_image is not None:
            mouth_extractor = MouthExtractor(config=self.config)
            open_rgb = load_rgb(open_mouth_image)
            closed_mask, open_mask = mouth_extractor.extract_pair(
                rgb, open_rgb
            )
            closed_rgba = _mask_to_rgba(rgb, closed_mask)
            open_rgba = _mask_to_rgba(open_rgb, open_mask)

            mouth_closed_path = output_dir / "mouth_closed.png"
            mouth_open_path = output_dir / "mouth_open.png"
            save_rgba(closed_rgba, mouth_closed_path)
            save_rgba(open_rgba, mouth_open_path)

        elapsed = time.perf_counter() - start
        logger.info(f"Stage 2 done in {elapsed:.1f}s")

        return Stage2Output(
            body_png=body_path,
            hair_front_png=hair_front_path,
            hair_back_png=hair_back_path,
            mouth_closed_png=mouth_closed_path,
            mouth_open_png=mouth_open_path,
            elapsed_seconds=elapsed,
        )


def _mask_to_rgba(rgb: "object", mask: "object") -> "object":
    """HxWx3 RGB + HxW bool mask → HxWx4 RGBA."""
    import numpy as np

    rgb_arr = np.asarray(rgb)
    mask_arr = np.asarray(mask).astype(bool)
    alpha = (mask_arr.astype(np.uint8) * 255)[..., None]
    return np.concatenate([rgb_arr, alpha], axis=-1).astype(np.uint8)


def _apply_mask_to_rgb(
    rgb: "object",
    mask: "object",
    base: "object | None",
) -> "object":
    """원본 RGB를 SAM3 마스크로 잘라 RGBA 생성.

    base가 주어지면 기존 레이어와 합성하여 경계선 보정 효과를 준다.
    현재는 단순 대체 구현. 추후 alpha blending 실험 대상.
    """
    import numpy as np

    cut = _mask_to_rgba(rgb, mask)
    if base is None:
        return cut

    base_arr = np.asarray(base)
    if base_arr.shape != cut.shape:
        logger.warning(
            f"Shape mismatch on neck blend: base={base_arr.shape} "
            f"vs cut={cut.shape} — falling back to raw cut"
        )
        return cut

    # 마스크 영역은 cut으로, 나머지는 base 유지.
    mask_arr = np.asarray(mask).astype(bool)[..., None]
    out = np.where(mask_arr, cut, base_arr)
    return out.astype(np.uint8)


def run_stage2(
    original_image: Path,
    psd_path: Path,
    output_dir: Path,
    open_mouth_image: Path | None = None,
    config: Stage2Config | None = None,
) -> Stage2Output:
    """Stage 2 편의 래퍼."""
    cfg = config or get_config().stage2
    pipeline = Stage2Pipeline(config=cfg)
    return pipeline(original_image, psd_path, output_dir, open_mouth_image)
