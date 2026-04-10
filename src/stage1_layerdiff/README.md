# Stage 1: See-Through MLX 포팅

## 현재 구조 (스캐폴딩)

```
stage1_layerdiff/
├── __init__.py         # 공개 API: Stage1Pipeline, run_stage1
├── inference.py        # 파이프라인 오케스트레이션 (스켈레톤)
├── model.py            # LayerDiffuseMLX 래퍼 (NotImplementedError)
├── marigold.py         # MarigoldMLX 래퍼 (NotImplementedError)
└── weights.py          # PyTorch → MLX 변환 유틸 (TODO)
```

## 상태

- 공용 인터페이스(`Stage1Pipeline`, `run_stage1`)는 확정됨.
- 실제 MLX 호출부는 모두 `NotImplementedError` 상태.
- Stage 2/3은 이 모듈의 PSD 출력을 입력으로 기대 → 상호 의존 최소.

## 시작 전 읽을 것

1. `wiki/see-through-porting.md`
2. MLX Stable Diffusion 예제: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion
3. See-Through 원본: `inference/scripts/inference_psd.py`

## 다음 작업

1. See-Through 레포 클론 후 `src/layerdiff/models/` 구조 파악
2. `model.py`에 SDXL UNet MLX 구현
3. `weights.py`로 가중치 변환 스크립트 완성
4. `inference.py`의 `decompose()` 채우기
5. `compute_iou()`로 CUDA 출력과 비교 검증
