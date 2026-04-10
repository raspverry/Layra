"""눈 깜빡임 프레임 생성."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.common.image_io import save_rgb
from src.common.logging import get_logger
from src.stage3_rife.interpolator import RIFEInterpolator

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


def generate_eye_blink_frames(
    eye_open: np.ndarray,
    eye_closed: np.ndarray,
    output_dir: Path,
    interpolator: RIFEInterpolator,
    n_frames: int = 8,
) -> list[Path]:
    """열림 ↔ 닫힘 프레임 시퀀스를 생성해 PNG로 저장.

    Args:
        eye_open: HxWx3 uint8 — 눈 완전 열린 상태.
        eye_closed: HxWx3 uint8 — 눈 완전 닫힌 상태.
        output_dir: 출력 디렉토리 (frame_001.png ~ frame_N.png).
        interpolator: 로드된 RIFEInterpolator 인스턴스.
        n_frames: 총 프레임 수 (시작/끝 포함).

    Returns:
        저장된 PNG 경로 리스트.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = interpolator.interpolate(eye_open, eye_closed, n_frames)

    paths: list[Path] = []
    for idx, frame in enumerate(frames, start=1):
        out = output_dir / f"frame_{idx:03d}.png"
        save_rgb(frame, out)
        paths.append(out)

    logger.info(f"Wrote {len(paths)} eye blink frames to {output_dir}")
    return paths
