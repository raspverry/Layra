"""SAM3 기반 목(neck) 영역 추출.

See-Through의 neck 레이어는 outpainting으로 생성되어 경계가 부자연스럽다.
이 모듈은 원본 이미지에서 SAM3으로 목 영역을 다시 마스킹해
깨끗한 경계의 body/neck 합성 이미지를 만든다.

SAM3 로드는 lazy하게 하여 Stage 2가 로드되는 것만으로는
3.2GB 체크포인트를 읽지 않도록 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.config import Stage2Config
from src.common.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass(slots=True)
class NeckExtractor:
    """SAM3 wrapper for neck region extraction."""

    config: Stage2Config

    def __post_init__(self) -> None:
        self._predictor = None

    def _ensure_loaded(self) -> None:
        """SAM3 모델을 최초 호출 시 로드."""
        if self._predictor is not None:
            return

        weights = self.config.sam3_weights
        if not weights.exists():
            raise FileNotFoundError(
                f"SAM3 weights not found: {weights}\n"
                "Download from "
                "https://github.com/facebookresearch/segment-anything-3"
            )

        # SAM3 (segment_anything_3)은 PyPI에 없어 git+ 설치 필요.
        # import를 함수 안에서 하여 Stage 2 import 자체는 SAM3 없어도 되게 함.
        try:
            from segment_anything_3 import SAM3, SamPredictor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "segment_anything_3 not installed. Run:\n"
                "  pip install git+https://github.com/"
                "facebookresearch/segment-anything-3.git"
            ) from e

        logger.info(f"Loading SAM3 from {weights}")
        model = SAM3()
        model.load_weights(str(weights))
        self._predictor = SamPredictor(model)

    def extract(
        self,
        original_image: "np.ndarray",
        point_hint: tuple[int, int] | None = None,
    ) -> "np.ndarray":
        """원본 이미지에서 목 영역 마스크를 추출.

        Args:
            original_image: HxWx3 uint8 numpy 배열 (See-Through 입력과 동일).
            point_hint: 목 위치 힌트 (x, y). None이면 자동 추정.

        Returns:
            HxW bool numpy 배열 (목 영역 마스크).
        """
        import numpy as np

        self._ensure_loaded()
        assert self._predictor is not None

        self._predictor.set_image(original_image)

        if point_hint is None:
            point_hint = self._estimate_neck_point(original_image)
            logger.debug(f"Estimated neck point: {point_hint}")

        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([point_hint]),
            point_labels=np.array([1]),
            multimask_output=self.config.multimask_output,
        )
        best = masks[int(np.argmax(scores))]
        return best.astype(bool)

    def _estimate_neck_point(
        self, image: "np.ndarray"
    ) -> tuple[int, int]:
        """얼굴 bbox가 없을 때의 단순 목 위치 추정.

        전신 포트레이트 일러스트 기준, 이미지 높이의 약 45% 지점을
        목으로 가정한다. 향후 face detection 통합 검토.
        """
        h, w = image.shape[:2]
        return (w // 2, int(h * self.config.neck_point_ratio_y))


def save_mask(mask: "np.ndarray", path: Path) -> None:
    """바이너리 마스크를 8-bit PNG로 저장 (디버깅용)."""
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(path)
