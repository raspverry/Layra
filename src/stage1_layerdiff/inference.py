"""Stage 1 추론 파이프라인 스켈레톤.

See-Through의 inference_psd.py를 MLX로 포팅한 최종 형태가 될 모듈.
현재는 파이프라인 구조와 인터페이스만 정의되어 있고,
실제 MLX 모델 호출은 TODO로 남겨져 있다.

맥북 도착 후 작업 순서:
    1. model.py에 LayerDiffuse MLX UNet 구현
    2. marigold.py에 Marigold MLX depth 모델 구현
    3. weights.py로 PyTorch 체크포인트 → MLX npz 변환
    4. 이 파일의 Stage1Pipeline.__call__ 구현
    5. wiki/see-through-porting.md Step 6 검증 (IoU)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.config import Stage1Config, get_config
from src.common.logging import get_logger
from src.common.psd_io import write_psd
from src.common.types import LayerSet, Stage1Output

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass(slots=True)
class Stage1Pipeline:
    """LayerDiffuse + Marigold MLX 파이프라인.

    맥북 도착 후 실제 모델 호출 구현 예정.
    """

    config: Stage1Config

    def __post_init__(self) -> None:
        self._layerdiff = None
        self._marigold = None

    def load_models(self) -> None:
        """LayerDiffuse와 Marigold 가중치를 MLX로 로드.

        Raises:
            FileNotFoundError: 가중치가 없을 때.
            NotImplementedError: MLX 포팅 미완료.
        """
        # TODO(stage1): model.py의 LayerDiffuseMLX 로드
        # TODO(stage1): marigold.py의 MarigoldMLX 로드
        raise NotImplementedError(
            "MLX porting pending — see wiki/see-through-porting.md"
        )

    def decompose(self, image: "np.ndarray") -> LayerSet:
        """단일 RGB 이미지 → 레이어 분해.

        Args:
            image: HxWx3 uint8 numpy 배열.

        Returns:
            LayerSet (레이어별 RGBA + drawing order).

        Raises:
            NotImplementedError: MLX 포팅 미완료.
        """
        if self._layerdiff is None:
            self.load_models()
        # TODO(stage1): LayerDiffuse sampling loop
        # TODO(stage1): Marigold depth → drawing order
        raise NotImplementedError(
            "MLX porting pending — see wiki/see-through-porting.md"
        )

    def __call__(
        self,
        image_path: Path,
        psd_output: Path,
    ) -> Stage1Output:
        """파이프라인 실행.

        Args:
            image_path: 입력 일러스트 경로.
            psd_output: PSD 출력 경로.

        Returns:
            Stage1Output (PSD 경로 + 경과 시간).
        """
        from src.common.image_io import load_rgb

        start = time.perf_counter()
        logger.info(f"Stage 1 start: {image_path}")

        rgb = load_rgb(image_path)
        layer_set = self.decompose(rgb)

        psd_path = write_psd(layer_set, psd_output)
        elapsed = time.perf_counter() - start
        logger.info(
            f"Stage 1 done: {len(layer_set)} layers in {elapsed:.1f}s"
        )

        return Stage1Output(
            layer_set=layer_set,
            psd_path=psd_path,
            elapsed_seconds=elapsed,
        )


def run_stage1(
    image_path: Path,
    psd_output: Path,
    config: Stage1Config | None = None,
) -> Stage1Output:
    """Stage 1 편의 래퍼.

    Args:
        image_path: 입력 이미지.
        psd_output: PSD 출력 경로.
        config: 덮어쓸 설정 (None이면 get_config() 사용).

    Returns:
        Stage1Output.
    """
    cfg = config or get_config().stage1
    pipeline = Stage1Pipeline(config=cfg)
    return pipeline(image_path, psd_output)
