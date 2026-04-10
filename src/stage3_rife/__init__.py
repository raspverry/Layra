"""Stage 3 — RIFE ONNX 기반 프레임 보간.

눈 깜빡임(열림 ↔ 닫힘) 및 입 모양 5모음(あいうえお)에 대해
중간 프레임을 생성한다. ONNX Runtime + CoreML(Mac) 또는 CPU fallback.
"""

from src.stage3_rife.eye_blink import generate_eye_blink_frames
from src.stage3_rife.interpolator import RIFEInterpolator, RIFEInterpolatorLike
from src.stage3_rife.mouth_frames import VOWELS, generate_mouth_frames
from src.stage3_rife.pipeline import Stage3Pipeline, run_stage3

__all__ = [
    "VOWELS",
    "RIFEInterpolator",
    "RIFEInterpolatorLike",
    "Stage3Pipeline",
    "generate_eye_blink_frames",
    "generate_mouth_frames",
    "run_stage3",
]
