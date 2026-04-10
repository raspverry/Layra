"""pytest 공용 fixture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_out_dir(tmp_path: Path) -> Iterator[Path]:
    """임시 출력 디렉토리."""
    d = tmp_path / "out"
    d.mkdir(parents=True, exist_ok=True)
    yield d


@pytest.fixture
def dummy_rgba_64x48() -> object:
    """64x48 RGBA 이미지 (왼쪽 절반 빨강, 오른쪽 절반 투명)."""
    import numpy as np

    arr = np.zeros((48, 64, 4), dtype=np.uint8)
    arr[:, :32, 0] = 255  # 빨강
    arr[:, :32, 3] = 255  # 불투명
    return arr


@pytest.fixture
def dummy_rgb_64x48() -> object:
    """64x48 RGB 이미지 (그라디언트)."""
    import numpy as np

    arr = np.zeros((48, 64, 3), dtype=np.uint8)
    for y in range(48):
        arr[y, :, 0] = int(y / 48 * 255)
        arr[y, :, 1] = 128
        arr[y, :, 2] = 255 - int(y / 48 * 255)
    return arr
