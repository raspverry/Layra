"""Stage 2 — SAM3로 See-Through PSD 레이어를 정밀 보정.

See-Through 단독으로는 목(neck)과 입(mouth) 레이어 경계가 부자연스럽다.
이 stage는 원본 이미지 + PSD 레이어를 입력받아, SAM3의 text-prompt 방식
(`Sam3Processor.set_text_prompt`)으로 경계선을 재추출해 자연스러운
body/hair/mouth PNG를 생성한다.

Text-prompt 방식은 PachiPakuGen `scripts/extract_neck_mask.py`와 동일하며
Point-prompt 방식보다 robust하다. 자세한 내용은 `wiki/sam3-rife.md` 참조.
"""

from src.stage2_sam3.mouth_extractor import MouthExtractor
from src.stage2_sam3.neck_extractor import NeckExtractor
from src.stage2_sam3.pipeline import Stage2Pipeline, run_stage2
from src.stage2_sam3.psd_parser import (
    extract_body_parts,
    layerset_to_body_parts,
)
from src.stage2_sam3.sam3_backend import (
    Sam3ProcessorLike,
    Sam3TextExtractor,
    combine_masks,
    extract_with_processor,
    postprocess_mask,
)

__all__ = [
    "MouthExtractor",
    "NeckExtractor",
    "Sam3ProcessorLike",
    "Sam3TextExtractor",
    "Stage2Pipeline",
    "combine_masks",
    "extract_body_parts",
    "extract_with_processor",
    "layerset_to_body_parts",
    "postprocess_mask",
    "run_stage2",
]
