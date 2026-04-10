# Stage 1: See-Through MLX 포팅

## 구조 (예정)

```
stage1_layerdiff/
├── __init__.py
├── model.py          # LayerDiffuse MLX 모델
├── marigold.py       # Marigold MLX 모델
├── inference.py      # 추론 파이프라인
├── weights.py        # PyTorch → MLX weights 변환
└── utils.py          # 유틸리티
```

## 시작 전 읽을 것

1. `wiki/see-through-porting.md`
2. MLX Stable Diffusion 예제: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion
3. See-Through 원본: `inference/scripts/inference_psd.py`
