"""PSD 입출력 — psd-tools 기반.

Stage 1 결과를 PSD로 저장하고, Stage 2 입력으로 다시 파싱한다.
레이어 이름은 LayerName StrEnum에 매핑된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import LayerName, LayerSet

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


def parse_psd(psd_path: Path | str) -> LayerSet:
    """See-Through 출력 PSD를 LayerSet으로 파싱.

    레이어 이름은 대소문자 무시하고 LayerName에 매칭을 시도한다.
    매칭 실패한 레이어는 경고 후 무시된다.

    Args:
        psd_path: PSD 파일 경로.

    Returns:
        LayerSet (drawing_order는 PSD의 bottom-to-top 순서).

    Raises:
        FileNotFoundError: PSD가 없을 때.
    """
    import numpy as np
    from psd_tools import PSDImage

    p = Path(psd_path)
    if not p.exists():
        raise FileNotFoundError(f"PSD not found: {p}")

    psd = PSDImage.open(p)
    layer_set = LayerSet()

    name_map = {name.value.lower(): name for name in LayerName}

    for layer in psd.descendants():
        if layer.is_group():
            continue

        raw_name = layer.name.strip().lower()
        matched = name_map.get(raw_name)
        if matched is None:
            logger.warning(f"Unknown layer name skipped: {layer.name!r}")
            continue

        composite = layer.composite()
        if composite is None:
            logger.warning(f"Layer {raw_name!r} has empty composite")
            continue

        rgba = np.asarray(composite.convert("RGBA"), dtype=np.uint8)
        layer_set.layers[matched] = rgba
        layer_set.drawing_order.append(matched)

    logger.info(f"Parsed {len(layer_set)} layers from {p.name}")
    return layer_set


def write_psd(
    layer_set: LayerSet,
    psd_path: Path | str,
    canvas_size: tuple[int, int] | None = None,
) -> Path:
    """LayerSet을 per-layer PSD로 저장.

    psd_tools 1.14+의 `PSDImage.new` + `create_pixel_layer` + `append`를
    사용하여 각 레이어가 분리된 PSD를 작성한다. `parse_psd`와 round-trip
    가능하다.

    Args:
        layer_set: 저장할 LayerSet. `drawing_order`가 비어 있으면 `layers`
            딕셔너리 삽입 순서를 사용한다.
        psd_path: 출력 경로.
        canvas_size: (width, height). None이면 첫 레이어 크기 사용.

    Returns:
        저장된 PSD 경로.
    """
    from PIL import Image
    from psd_tools import PSDImage

    if not layer_set.layers:
        raise ValueError("Cannot write empty LayerSet")

    if canvas_size is None:
        first = next(iter(layer_set.layers.values()))
        canvas_size = (first.shape[1], first.shape[0])

    p = Path(psd_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    psd = PSDImage.new(mode="RGBA", size=canvas_size)
    order = layer_set.drawing_order or list(layer_set.layers.keys())
    for name in order:
        rgba = layer_set.layers.get(name)
        if rgba is None:
            continue
        pil = Image.fromarray(rgba, mode="RGBA")
        layer = psd.create_pixel_layer(pil, name=name.value)
        psd.append(layer)

    psd.save(p)
    logger.info(f"Wrote PSD with {len(order)} layers to {p}")
    return p


def _alpha_composite(
    base: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    """HxWx4 uint8 배열 두 개를 over-compositing.

    base, overlay는 같은 크기여야 한다.
    """
    import numpy as np

    if base.shape != overlay.shape:
        raise ValueError(f"Shape mismatch: base={base.shape} overlay={overlay.shape}")

    base_f = base.astype(np.float32) / 255.0
    over_f = overlay.astype(np.float32) / 255.0

    a_over = over_f[..., 3:4]
    a_base = base_f[..., 3:4]
    a_out = a_over + a_base * (1.0 - a_over)

    rgb_out = over_f[..., :3] * a_over + base_f[..., :3] * a_base * (1.0 - a_over)
    # a_out이 0에 가까운 곳은 나누지 말고 0으로.
    safe = np.where(a_out > 1e-6, a_out, 1.0)
    rgb_out = rgb_out / safe

    out = np.concatenate([rgb_out, a_out], axis=-1)
    return (out * 255.0).clip(0, 255).astype(np.uint8)
