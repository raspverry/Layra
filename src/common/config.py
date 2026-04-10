"""프로젝트 전역 설정.

환경 변수와 기본값을 Pydantic 모델로 관리한다.
실제 모델 경로, 디바이스, 하이퍼파라미터는 여기서 일원화해서 읽는다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Device = Literal["mlx", "mps", "cpu", "cuda"]


class Stage1Config(BaseModel):
    """Stage 1 (LayerDiffuse + Marigold) 설정."""

    layerdiff_weights: Path = Field(
        default=Path("models/layerdiff"),
        description="LayerDiffuse MLX weights 디렉토리",
    )
    marigold_weights: Path = Field(
        default=Path("models/marigold"),
        description="Marigold MLX weights 디렉토리",
    )
    resolution: int = Field(
        default=1280,
        description="추론 해상도. 원본 See-Through는 1280 기준.",
    )
    num_inference_steps: int = Field(
        default=30,
        description="Diffusion sampling step 수",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="Classifier-free guidance scale",
    )
    dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="MLX precision. M5 Pro는 bfloat16 권장.",
    )
    max_layers: int = Field(
        default=23,
        description="See-Through 최대 레이어 수",
    )


class Stage2Config(BaseModel):
    """Stage 2 (SAM3) 설정."""

    sam3_weights: Path = Field(
        default=Path("models/sam3_vit_h.pth"),
        description="SAM3 체크포인트 경로",
    )
    bpe_path: Path | None = Field(
        default=None,
        description=(
            "SAM3 BPE 토크나이저 경로. None이면 sam3 패키지 기본값 "
            "(sam3/assets/bpe_simple_vocab_16e6.txt.gz)을 사용."
        ),
    )
    device: Device = Field(
        default="mps",
        description="SAM3 실행 디바이스 (Mac은 mps)",
    )

    # --- Text-prompt based extraction (PachiPakuGen 스타일) ---
    confidence_threshold: float = Field(
        default=0.3,
        description="Sam3Processor confidence threshold (PachiPakuGen 값)",
    )
    text_prompts: dict[str, str] = Field(
        default_factory=lambda: {
            "neck": "neck",
            "mouth": "mouth",
            "eye": "eye",
        },
        description=(
            "파트 이름 → SAM3 text prompt 매핑. 단일 단어 또는 콤마 구분 리스트"
            "(e.g. 'eye,eyelid'). 기본값은 PachiPakuGen과 동일."
        ),
    )

    # --- Mask postprocessing ---
    postprocess_dilate_iterations: int = Field(
        default=2,
        description="바이너리 마스크 dilate 반복 수 (경계선 부드럽게)",
    )
    postprocess_blur_kernel: int = Field(
        default=7,
        description="Gaussian blur 커널 크기 (홀수). 0이면 블러 비활성화.",
    )

    # --- Fallback (point-based) ---
    neck_point_ratio_y: float = Field(
        default=0.45,
        description=(
            "Text-prompt 방식이 실패하거나 fallback이 필요할 때 쓰는 "
            "y축 비율. (deprecated: 주로 text prompt를 사용)"
        ),
    )
    multimask_output: bool = Field(
        default=True,
        description="legacy point-based 경로의 SAM3 multimask 사용 여부",
    )


class Stage3Config(BaseModel):
    """Stage 3 (RIFE) 설정."""

    rife_model: Path = Field(
        default=Path("models/rife.onnx"),
        description="RIFE ONNX 모델 경로",
    )
    providers: list[str] = Field(
        default_factory=lambda: [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
        description="ONNX Runtime execution provider 우선순위",
    )
    eye_blink_frames: int = Field(
        default=8,
        description="눈 깜빡임 중간 프레임 수",
    )
    mouth_frames: int = Field(
        default=8,
        description="입 모양 중간 프레임 수 (모음당)",
    )


class LayraConfig(BaseModel):
    """Layra 전체 설정."""

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
    )
    model_dir: Path = Field(
        default=Path("models"),
        description="모델 가중치 루트 디렉토리",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="결과물 출력 루트 디렉토리",
    )
    log_level: str = Field(default="INFO")

    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    stage3: Stage3Config = Field(default_factory=Stage3Config)

    def resolve_paths(self) -> LayraConfig:
        """상대 경로들을 project_root 기준 절대 경로로 변환한다."""
        root = self.project_root

        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else (root / p).resolve()

        self.model_dir = _abs(self.model_dir)
        self.output_dir = _abs(self.output_dir)
        self.stage1.layerdiff_weights = _abs(self.stage1.layerdiff_weights)
        self.stage1.marigold_weights = _abs(self.stage1.marigold_weights)
        self.stage2.sam3_weights = _abs(self.stage2.sam3_weights)
        self.stage3.rife_model = _abs(self.stage3.rife_model)
        return self


@lru_cache(maxsize=1)
def get_config() -> LayraConfig:
    """환경 변수 + 기본값으로 LayraConfig 생성.

    지원 환경 변수:
        LAYRA_MODEL_DIR, LAYRA_OUTPUT_DIR, LAYRA_LOG_LEVEL,
        SAM3_MODEL_PATH, RIFE_MODEL_PATH
    """
    cfg = LayraConfig()

    if env_model_dir := os.getenv("LAYRA_MODEL_DIR"):
        cfg.model_dir = Path(env_model_dir)
    if env_output_dir := os.getenv("LAYRA_OUTPUT_DIR"):
        cfg.output_dir = Path(env_output_dir)
    if env_log_level := os.getenv("LAYRA_LOG_LEVEL"):
        cfg.log_level = env_log_level
    if env_sam3 := os.getenv("SAM3_MODEL_PATH"):
        cfg.stage2.sam3_weights = Path(env_sam3)
    if env_rife := os.getenv("RIFE_MODEL_PATH"):
        cfg.stage3.rife_model = Path(env_rife)

    return cfg.resolve_paths()
