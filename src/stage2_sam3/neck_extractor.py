"""SAM3 기반 목(neck) 영역 추출 — text prompt 방식.

See-Through의 neck 레이어는 outpainting으로 생성되어 경계가 부자연스럽다.
이 모듈은 원본 이미지에서 SAM3으로 목 영역을 text prompt("neck")로 추출해
깨끗한 경계의 마스크를 반환한다.

구현은 PachiPakuGen의 `extract_neck_mask.py`를 따르되, Layra는
`Sam3TextExtractor` 백엔드를 재사용하여 Mouth/Eye와 로직을 공유한다.

사용 예::

    extractor = NeckExtractor(config=stage2_cfg)
    mask = extractor.extract(image_rgb)  # HxW uint8

레거시 point-based 호출은 `extract_from_point(image, point)`로 남겨두지만
기본 경로는 text prompt다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.common.config import Stage2Config
from src.common.logging import get_logger
from src.stage2_sam3.sam3_backend import Sam3TextExtractor

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass(slots=True)
class NeckExtractor:
    """Text-prompt based neck mask extractor."""

    config: Stage2Config
    _backend: Sam3TextExtractor | None = field(default=None, init=False, repr=False)

    def _backend_or_build(self) -> Sam3TextExtractor:
        if self._backend is None:
            self._backend = Sam3TextExtractor(config=self.config)
        return self._backend

    def inject_backend(self, backend: Sam3TextExtractor) -> None:
        """Override the backend (used by tests)."""
        self._backend = backend

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract neck mask from an RGB image.

        Args:
            image: HxWx3 uint8 numpy array (same as Stage 1 input).

        Returns:
            HxW uint8 mask (0..255) with dilate+blur post-processing.
        """
        return self._backend_or_build().extract_named(image, "neck")

    def extract_from_point(
        self,
        image: np.ndarray,
        point_hint: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Legacy point-based fallback.

        Kept only for environments where text-prompt SAM3 is unavailable
        or returns empty results. Prefer `extract()` which uses the
        PachiPakuGen-style text prompt flow.
        """
        import numpy as np

        backend = self._backend_or_build()
        backend._ensure_loaded()  # type: ignore[attr-defined]
        assert backend._processor is not None  # type: ignore[attr-defined]

        if point_hint is None:
            point_hint = self._estimate_neck_point(image)
            logger.debug(f"Estimated neck point: {point_hint}")

        # Some SAM3 builds expose a lower-level `predict` with point prompts.
        # We keep this path as best-effort — it is exercised by tests via mocks.
        predictor = backend._processor  # type: ignore[attr-defined]
        if not hasattr(predictor, "predict"):
            raise NotImplementedError(
                "Installed SAM3 processor does not expose a point-based "
                "`predict` API. Use NeckExtractor.extract() (text prompt)."
            )

        masks, scores, _ = predictor.predict(  # type: ignore[attr-defined]
            point_coords=np.array([point_hint]),
            point_labels=np.array([1]),
            multimask_output=self.config.multimask_output,
        )
        best: np.ndarray = masks[int(np.argmax(scores))].astype(bool)
        return best

    def _estimate_neck_point(self, image: np.ndarray) -> tuple[int, int]:
        h, w = image.shape[:2]
        return (w // 2, int(h * self.config.neck_point_ratio_y))
