"""이미지 I/O 유틸 — Pillow + numpy 기반.

Layra는 이미지를 HxWxC uint8 numpy 배열로 표준화한다.
알파 채널은 RGBA 전체 파이프라인에서 유지된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def load_rgba(path: Path | str) -> np.ndarray:
    """이미지를 RGBA 형식으로 로드.

    Args:
        path: 이미지 파일 경로 (PNG / JPG / BMP 등 Pillow 지원 포맷).

    Returns:
        HxWx4 uint8 numpy 배열.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    import numpy as np
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    img = Image.open(p).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def load_rgb(path: Path | str) -> np.ndarray:
    """이미지를 RGB 형식으로 로드 (알파 무시).

    Args:
        path: 이미지 파일 경로.

    Returns:
        HxWx3 uint8 numpy 배열.
    """
    import numpy as np
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def save_rgba(array: np.ndarray, path: Path | str) -> None:
    """HxWx4 uint8 배열을 PNG로 저장.

    Args:
        array: HxWx4 uint8 numpy 배열.
        path: 출력 파일 경로. 부모 디렉토리는 자동 생성.

    Raises:
        ValueError: 배열 shape/dtype이 맞지 않을 때.
    """
    import numpy as np
    from PIL import Image

    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 array, got shape={array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 array, got dtype={array.dtype}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGBA").save(p)


def save_rgb(array: np.ndarray, path: Path | str) -> None:
    """HxWx3 uint8 배열을 PNG/JPG로 저장."""
    import numpy as np
    from PIL import Image

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 array, got shape={array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 array, got dtype={array.dtype}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(p)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """두 바이너리 마스크 간 Intersection-over-Union.

    포팅 검증 (Stage 1 CUDA vs MLX) 및 SAM3 품질 평가에 사용.

    Args:
        mask_a: HxW 배열 (bool 또는 {0, 1}).
        mask_b: HxW 배열 (bool 또는 {0, 1}).

    Returns:
        IoU ∈ [0, 1]. 두 마스크 모두 빈 경우 1.0 반환.
    """
    import numpy as np

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)
