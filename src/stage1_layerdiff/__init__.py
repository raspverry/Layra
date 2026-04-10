"""Stage 1 — LayerDiffuse + Marigold MLX 포팅.

단일 이미지로부터 최대 23개의 의미론적 RGBA 레이어를 생성한다.
원본: https://github.com/shitagaki-lab/see-through (Apache 2.0)
"""

from src.stage1_layerdiff.inference import Stage1Pipeline, run_stage1

__all__ = ["Stage1Pipeline", "run_stage1"]
