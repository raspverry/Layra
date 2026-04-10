"""PSD → Stage 2 입력 변환.

`src/common/psd_io.py`의 `parse_psd`를 사용해 PSD를 LayerSet으로 읽고,
Stage 2에서 필요한 body / hair_front / hair_back을 추출한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import LayerName, LayerSet

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass(slots=True)
class BodyParts:
    """Stage 2가 조작할 대상 파트."""

    body: np.ndarray | None
    hair_front: np.ndarray | None
    hair_back: np.ndarray | None
    mouth: np.ndarray | None
    neck: np.ndarray | None


def layerset_to_body_parts(layer_set: LayerSet) -> BodyParts:
    """LayerSet에서 Stage 2가 필요로 하는 레이어만 추출.

    See-Through 원본은 hair를 front/back으로 분리하지만,
    단일 hair 레이어만 있는 경우(엣지 케이스)도 대응한다.

    Args:
        layer_set: Stage 1 출력 LayerSet.

    Returns:
        BodyParts — 누락된 파트는 None.
    """
    body = layer_set.get(LayerName.BODY)
    hair_front = layer_set.get(LayerName.HAIR_FRONT)
    hair_back = layer_set.get(LayerName.HAIR_BACK)
    mouth = layer_set.get(LayerName.MOUTH)
    neck = layer_set.get(LayerName.NECK)

    if hair_front is None and hair_back is None:
        logger.warning(
            "Neither hair_front nor hair_back present — "
            "Stage 1 output may need depth-based hair split"
        )

    if body is None:
        logger.warning("Body layer missing from Stage 1 output")

    return BodyParts(
        body=body,
        hair_front=hair_front,
        hair_back=hair_back,
        mouth=mouth,
        neck=neck,
    )


def extract_body_parts(
    layer_set: LayerSet,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """`layerset_to_body_parts`의 간단한 3-tuple 버전.

    Returns:
        (body, hair_front, hair_back) — 각 값은 None 가능.
    """
    parts = layerset_to_body_parts(layer_set)
    return parts.body, parts.hair_front, parts.hair_back
