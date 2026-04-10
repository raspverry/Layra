"""SAM3 기반 입(mouth) 영역 추출.

See-Through의 mouth 레이어는 정밀도가 낮은 경우가 많다.
이 모듈은 사용자가 별도로 제공한 '열린 입' 이미지와 원본의 '닫힌 입'에서
각각 SAM3으로 마스크를 추출해 Stage 3 RIFE의 입력 쌍을 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.common.config import Stage2Config
from src.common.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass(slots=True)
class MouthExtractor:
    """SAM3 wrapper for mouth region extraction."""

    config: Stage2Config

    def __post_init__(self) -> None:
        self._predictor = None

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return

        try:
            from segment_anything_3 import SAM3, SamPredictor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "segment_anything_3 not installed. Run:\n"
                "  pip install git+https://github.com/"
                "facebookresearch/segment-anything-3.git"
            ) from e

        weights = self.config.sam3_weights
        if not weights.exists():
            raise FileNotFoundError(f"SAM3 weights not found: {weights}")

        logger.info(f"Loading SAM3 from {weights}")
        model = SAM3()
        model.load_weights(str(weights))
        self._predictor = SamPredictor(model)

    def extract_pair(
        self,
        closed_image: "np.ndarray",
        open_image: "np.ndarray",
        closed_point: tuple[int, int] | None = None,
        open_point: tuple[int, int] | None = None,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """닫힌 입 / 열린 입 한 쌍의 마스크 추출.

        Args:
            closed_image: HxWx3 원본 이미지 (닫힌 입).
            open_image: HxWx3 사용자 제공 이미지 (열린 입).
            closed_point: 닫힌 입 위치 힌트 (x, y).
            open_point: 열린 입 위치 힌트 (x, y).

        Returns:
            (closed_mask, open_mask) — 둘 다 HxW bool.
        """
        closed = self._extract_single(closed_image, closed_point)
        opened = self._extract_single(open_image, open_point)
        return closed, opened

    def _extract_single(
        self,
        image: "np.ndarray",
        point_hint: tuple[int, int] | None,
    ) -> "np.ndarray":
        import numpy as np

        self._ensure_loaded()
        assert self._predictor is not None

        self._predictor.set_image(image)
        if point_hint is None:
            point_hint = self._estimate_mouth_point(image)

        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([point_hint]),
            point_labels=np.array([1]),
            multimask_output=self.config.multimask_output,
        )
        return masks[int(np.argmax(scores))].astype(bool)

    def _estimate_mouth_point(
        self, image: "np.ndarray"
    ) -> tuple[int, int]:
        """얼굴 검출 없을 때의 단순 입 위치 추정.

        이미지 중앙 수평, 높이 ~35% 지점을 입으로 가정.
        """
        h, w = image.shape[:2]
        return (w // 2, int(h * 0.35))
