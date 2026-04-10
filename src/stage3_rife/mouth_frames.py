"""입 모양 (5모음) 프레임 생성.

각 모음에 대해 '닫힘 → 모음' 시퀀스를 RIFE로 보간하여
Live2D Cubism에서 키프레임으로 사용할 수 있는 PNG 세트를 만든다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.common.image_io import save_rgb
from src.common.logging import get_logger
from src.stage3_rife.interpolator import RIFEInterpolator

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

VOWELS: tuple[str, ...] = ("a", "i", "u", "e", "o")
"""あいうえお 5모음."""


def generate_mouth_frames(
    mouth_closed: np.ndarray,
    mouth_shapes: dict[str, np.ndarray],
    output_dir: Path,
    interpolator: RIFEInterpolator,
    n_frames: int = 8,
) -> dict[str, list[Path]]:
    """5모음 각각에 대해 닫힘 → 모음 시퀀스 생성.

    Args:
        mouth_closed: HxWx3 uint8 — 닫힌 입 이미지.
        mouth_shapes: {"a": img, "i": img, ...} 형식의 모음별 열린 입.
            키는 VOWELS 중 하나여야 한다. 누락된 키는 경고 후 스킵.
        output_dir: 루트 출력 디렉토리. 하위에 mouth_a/, mouth_i/... 생성.
        interpolator: 로드된 RIFEInterpolator.
        n_frames: 모음당 총 프레임 수.

    Returns:
        {"a": [Path,...], "i": [...], ...} 형식의 경로 매핑.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}

    for vowel in VOWELS:
        if vowel not in mouth_shapes:
            logger.warning(f"Mouth shape for {vowel!r} missing — skipping")
            continue

        vowel_dir = output_dir / f"mouth_{vowel}"
        vowel_dir.mkdir(parents=True, exist_ok=True)

        frames = interpolator.interpolate(mouth_closed, mouth_shapes[vowel], n_frames)
        paths: list[Path] = []
        for idx, frame in enumerate(frames, start=1):
            out = vowel_dir / f"frame_{idx:03d}.png"
            save_rgb(frame, out)
            paths.append(out)
        result[vowel] = paths
        logger.info(f"Wrote {len(paths)} frames for mouth_{vowel}")

    return result
