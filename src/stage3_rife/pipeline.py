"""Stage 3 오케스트레이션."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import Stage3Config, get_config
from src.common.image_io import load_rgb
from src.common.logging import get_logger
from src.common.types import Stage3Output
from src.stage3_rife.eye_blink import generate_eye_blink_frames
from src.stage3_rife.interpolator import RIFEInterpolator, RIFEInterpolatorLike
from src.stage3_rife.mouth_frames import VOWELS, generate_mouth_frames

logger = get_logger(__name__)


@dataclass(slots=True)
class Stage3Inputs:
    """Stage 3 입력 파일 경로 묶음."""

    eye_open: Path
    eye_closed: Path
    mouth_closed: Path
    mouth_vowels: dict[str, Path]  # {"a": Path, ...}


@dataclass(slots=True)
class Stage3Pipeline:
    """RIFE 프레임 보간 파이프라인.

    `interpolator`를 주입하면 ONNX 세션 없이 테스트 가능 (mock용).
    주입하지 않으면 `config.rife_model`로 RIFEInterpolator를 lazy 생성.
    """

    config: Stage3Config
    interpolator: RIFEInterpolatorLike | None = field(default=None)

    def _get_interpolator(self) -> RIFEInterpolatorLike:
        if self.interpolator is None:
            self.interpolator = RIFEInterpolator(
                model_path=self.config.rife_model,
                providers=self.config.providers,
            )
        return self.interpolator

    def __call__(
        self,
        inputs: Stage3Inputs,
        output_dir: Path,
    ) -> Stage3Output:
        """Stage 3 실행.

        Args:
            inputs: 눈/입 쌍 경로 묶음.
            output_dir: 출력 루트 (eye/, mouth_a/, ...).

        Returns:
            Stage3Output.
        """
        start = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Stage 3 start: output_dir={output_dir}")

        interpolator = self._get_interpolator()

        eye_dir = output_dir / "eye"
        generate_eye_blink_frames(
            eye_open=load_rgb(inputs.eye_open),
            eye_closed=load_rgb(inputs.eye_closed),
            output_dir=eye_dir,
            interpolator=interpolator,
            n_frames=self.config.eye_blink_frames,
        )

        mouth_closed_img = load_rgb(inputs.mouth_closed)
        mouth_shapes = {
            vowel: load_rgb(path)
            for vowel, path in inputs.mouth_vowels.items()
            if vowel in VOWELS
        }
        mouth_paths = generate_mouth_frames(
            mouth_closed=mouth_closed_img,
            mouth_shapes=mouth_shapes,
            output_dir=output_dir,
            interpolator=interpolator,
            n_frames=self.config.mouth_frames,
        )

        mouth_dirs = {v: (output_dir / f"mouth_{v}") for v in mouth_paths}
        elapsed = time.perf_counter() - start
        logger.info(f"Stage 3 done in {elapsed:.1f}s")

        return Stage3Output(
            eye_frames_dir=eye_dir,
            mouth_frames_dirs=mouth_dirs,
            elapsed_seconds=elapsed,
        )


def run_stage3(
    inputs: Stage3Inputs,
    output_dir: Path,
    config: Stage3Config | None = None,
) -> Stage3Output:
    """Stage 3 편의 래퍼."""
    cfg = config or get_config().stage3
    pipeline = Stage3Pipeline(config=cfg)
    return pipeline(inputs, output_dir)
