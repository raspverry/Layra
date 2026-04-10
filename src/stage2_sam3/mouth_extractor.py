"""SAM3 기반 입(mouth) 영역 추출 — text prompt 방식.

Stage 3 RIFE의 '닫힌 입'/'열린 입' 쌍을 만들기 위해 원본 이미지와
사용자가 제공한 '열린 입' 이미지에서 각각 mouth 마스크를 추출한다.
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
class MouthExtractor:
    """Text-prompt based mouth mask extractor."""

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
        """Extract mouth mask from a single RGB image."""
        return self._backend_or_build().extract_named(image, "mouth")

    def extract_pair(
        self,
        closed_image: np.ndarray,
        open_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (closed_mouth_mask, open_mouth_mask) pair.

        Args:
            closed_image: HxWx3 uint8 — original input (mouth closed).
            open_image: HxWx3 uint8 — user-supplied image with mouth open.

        Returns:
            Two HxW uint8 (0..255) masks.
        """
        backend = self._backend_or_build()
        closed = backend.extract_named(closed_image, "mouth")
        opened = backend.extract_named(open_image, "mouth")
        return closed, opened
