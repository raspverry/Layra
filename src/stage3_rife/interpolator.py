"""RIFE ONNX Runtime 래퍼.

두 RGB 이미지 사이에 N개의 중간 프레임을 생성한다.
CoreML Provider를 우선 시도하고, 없으면 CPU로 fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass(slots=True)
class RIFEInterpolator:
    """RIFE ONNX 모델 래퍼.

    Attributes:
        model_path: .onnx 파일 경로.
        providers: ONNX Runtime execution provider 우선순위.
    """

    model_path: Path
    providers: list[str] = field(
        default_factory=lambda: [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    _session: object | None = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"RIFE ONNX model not found: {self.model_path}")

        import onnxruntime as ort

        available = ort.get_available_providers()
        selected = [p for p in self.providers if p in available]
        if not selected:
            logger.warning(
                f"None of {self.providers} available, falling back to {available}"
            )
            selected = available

        logger.info(f"Loading RIFE ONNX with providers={selected}")
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=selected,
        )

    def interpolate(
        self,
        frame_start: np.ndarray,
        frame_end: np.ndarray,
        n_frames: int = 8,
    ) -> list[np.ndarray]:
        """두 프레임 사이 N개의 중간 프레임 생성.

        Args:
            frame_start: HxWx3 uint8 시작 프레임.
            frame_end: HxWx3 uint8 끝 프레임.
            n_frames: 출력 시퀀스 총 프레임 수 (시작/끝 포함).

        Returns:
            uint8 frames 리스트 (길이 n_frames).
        """
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {n_frames}")

        self._ensure_loaded()
        frames: list[np.ndarray] = [frame_start]
        for i in range(1, n_frames - 1):
            t = i / (n_frames - 1)
            mid = self._interpolate_single(frame_start, frame_end, t)
            frames.append(mid)
        frames.append(frame_end)
        return frames

    def _interpolate_single(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        import numpy as np

        assert self._session is not None
        inputs = {
            "img0": self._preprocess(img0),
            "img1": self._preprocess(img1),
            "timestep": np.array([timestep], dtype=np.float32),
        }
        # 일부 RIFE ONNX 변형은 timestep 입력이 없을 수 있음.
        session_inputs = {
            i.name
            for i in self._session.get_inputs()  # type: ignore
        }
        inputs = {k: v for k, v in inputs.items() if k in session_inputs}
        output = self._session.run(None, inputs)  # type: ignore
        return self._postprocess(output[0])

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """HxWxC uint8 → 1xCxHxW float32 (0~1)."""
        import numpy as np

        arr = img.astype(np.float32) / 255.0
        return arr.transpose(2, 0, 1)[np.newaxis]

    @staticmethod
    def _postprocess(output: np.ndarray) -> np.ndarray:
        """1xCxHxW float32 (0~1) → HxWxC uint8."""
        import numpy as np

        arr = output[0].transpose(1, 2, 0)
        result: np.ndarray = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return result
