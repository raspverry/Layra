"""Layra 파이프라인 전역 데이터 타입.

Stage 간 데이터 전달에 쓰이는 공용 데이터 클래스를 정의한다.
실제 이미지 데이터는 numpy 배열(uint8, HxWxC)로 표준화한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    RGBAImage = np.ndarray
    """HxWx4 uint8 numpy 배열."""
else:
    RGBAImage = "np.ndarray"


class LayerName(StrEnum):
    """See-Through 23개 표준 레이어 이름.

    원본 논문의 23-layer 레이아웃을 StrEnum으로 표준화.
    """

    # --- Head region ---
    HAIR_FRONT = "hair_front"
    HAIR_BACK = "hair_back"
    HAIR_SIDE_L = "hair_side_L"
    HAIR_SIDE_R = "hair_side_R"
    FACE = "face"
    EYEBROW_L = "eyebrow_L"
    EYEBROW_R = "eyebrow_R"
    EYE_L = "eye_L"
    EYE_R = "eye_R"
    EYE_WHITE_L = "eye_white_L"
    EYE_WHITE_R = "eye_white_R"
    NOSE = "nose"
    MOUTH = "mouth"
    EAR_L = "ear_L"
    EAR_R = "ear_R"

    # --- Body region ---
    NECK = "neck"
    BODY = "body"
    ARM_L = "arm_L"
    ARM_R = "arm_R"
    HAND_L = "hand_L"
    HAND_R = "hand_R"
    CLOTHES = "clothes"
    ACCESSORIES = "accessories"


@dataclass(slots=True)
class LayerSet:
    """하나의 캐릭터에서 분해된 레이어 집합.

    Attributes:
        layers: 레이어 이름 → RGBA numpy 배열 매핑.
        drawing_order: 뒤에서 앞으로의 레이어 순서 (Marigold depth 기반).
    """

    layers: dict[LayerName, RGBAImage] = field(default_factory=dict)
    drawing_order: list[LayerName] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.layers)

    def __contains__(self, name: LayerName) -> bool:
        return name in self.layers

    def get(self, name: LayerName) -> RGBAImage | None:
        return self.layers.get(name)


@dataclass(slots=True)
class Stage1Output:
    """Stage 1 (레이어 분해) 출력."""

    layer_set: LayerSet
    psd_path: Path
    elapsed_seconds: float


@dataclass(slots=True)
class Stage2Output:
    """Stage 2 (SAM3 정밀 추출) 출력."""

    body_png: Path
    hair_front_png: Path
    hair_back_png: Path
    mouth_closed_png: Path | None
    mouth_open_png: Path | None
    elapsed_seconds: float


@dataclass(slots=True)
class Stage3Output:
    """Stage 3 (RIFE 프레임 보간) 출력."""

    eye_frames_dir: Path
    mouth_frames_dirs: dict[str, Path]  # {"a": Path, "i": Path, ...}
    elapsed_seconds: float


@dataclass(slots=True)
class PipelineResult:
    """전체 파이프라인 최종 결과."""

    input_image: Path
    output_dir: Path
    stage1: Stage1Output
    stage2: Stage2Output
    stage3: Stage3Output
    total_elapsed_seconds: float
