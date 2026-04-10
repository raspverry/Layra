"""Stage 2 — SAM3로 See-Through PSD 레이어를 정밀 보정.

See-Through 단독으로는 목(neck)과 입(mouth) 레이어 경계가 부자연스럽다.
이 stage는 원본 이미지 + PSD 레이어를 입력받아,
SAM3로 경계선을 재추출해 자연스러운 body/hair/mouth PNG를 생성한다.
"""

from src.stage2_sam3.mouth_extractor import MouthExtractor
from src.stage2_sam3.neck_extractor import NeckExtractor
from src.stage2_sam3.pipeline import Stage2Pipeline, run_stage2
from src.stage2_sam3.psd_parser import (
    extract_body_parts,
    layerset_to_body_parts,
)

__all__ = [
    "MouthExtractor",
    "NeckExtractor",
    "Stage2Pipeline",
    "extract_body_parts",
    "layerset_to_body_parts",
    "run_stage2",
]
