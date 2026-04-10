# 기술 파이프라인 상세

## Stage 1: See-Through (레이어 분해)

### 원본 구조 (CUDA)

```
inference_psd.py
    ├── LayerDiff 3D model (SDXL 기반)
    │     CUDA 12.8, bf16 precision
    │     1280 해상도 기준 12~16GB VRAM
    │     입력: RGB 이미지
    │     출력: 최대 23개 RGBA 레이어
    │
    └── Fine-tuned Marigold (depth estimation)
          입력: RGB 이미지
          출력: per-layer pseudo-depth
          → drawing order 결정에 사용
```

### 23개 레이어 구성

```
Head region:
  hair_front, hair_back, hair_side_L, hair_side_R
  face, eyebrow_L, eyebrow_R
  eye_L, eye_R, eye_white_L, eye_white_R
  nose, mouth, ear_L, ear_R

Body region:
  neck, body, arm_L, arm_R, hand_L, hand_R
  clothes, accessories
```

### MLX 포팅 전략

**포팅 대상**: LayerDiffuse + Marigold  
**포팅 미대상**: SAM3, RIFE (PyTorch MPS로 충분)

MLX에 이미 Stable Diffusion 예제 존재 → SDXL 기반 LayerDiffuse 포팅 경로 있음

```python
# CUDA 코드 패턴
import torch
model = model.to("cuda")
output = model(input.cuda())

# MLX 포팅 패턴
import mlx.core as mx
import mlx.nn as nn
# unified memory - .to() 불필요
output = model(input)
```

**주의사항**:
- CUDA 특화 ops (flash attention 등) → MLX 동등 ops로 교체 필요
- bf16 → MLX의 bfloat16 지원 확인 필요
- 커스텀 CUDA 커널 있으면 Metal shader로 재작성 필요

### 참조 명령어 (원본)

```bash
# 기본 실행 (12-16GB VRAM)
python inference/scripts/inference_psd.py \
  --srcp assets/test_image.png \
  --save_to_psd

# group offload (10GB로 줄임)
python inference/scripts/inference_psd.py \
  --srcp assets/test_image.png \
  --save_to_psd \
  --group_offload

# NF4 양자화 (8GB GPU용)
# NF4 파이프라인 별도 스크립트
```

---

## Stage 2: SAM3 + 정밀 추출

### 왜 필요한가

See-Through 단독 문제점:
- **목(neck)**: outpainting으로 생성 → 레이어 중첩 시 경계선 부자연스러움
- **입(mouth)**: mouth 레이어 검출 정밀도 부족

### SAM3 파이프라인

```
원본 이미지 (See-Through 입력과 동일)
    ↓
SAM3 (Segment Anything Model 3)
    point prompt or box prompt 입력
    → 목/입 영역 고정밀 마스크 생성
    ↓
See-Through PSD 레이어와 합성
    → 경계선 자연스럽게 보정
```

```python
# SAM3 사용 패턴 (PachiPakuGen 참조)
from sam3 import SAM3

model = SAM3()
# 목 추출
neck_mask = model.predict(
    image=original_image,
    point_coords=[[neck_x, neck_y]],
    point_labels=[1]
)
# 입 추출  
mouth_mask = model.predict(
    image=open_mouth_image,
    point_coords=[[mouth_x, mouth_y]],
    point_labels=[1]
)
```

**SAM3 모델 파일**: 약 3.2GB 별도 다운로드 필요

### 출력 파트

```
body.png      → 몸통 (목 포함, 정밀 추출)
hair.png      → 앞머리
hair_back.png → 뒷머리
```

---

## Stage 3: RIFE 프레임 보간

### 목적

정지 이미지 2장(닫힘/열림) 사이의 중간 프레임을 자동 생성

```
눈 깜빡임: 눈 열림 → 눈 닫힘 (N개 중간 프레임)
입 모양:   입 닫힘 ↔ あ/い/う/え/お (각 N개 중간 프레임)
```

### RIFE 파이프라인

```
frame_start.png (닫힘 상태)
frame_end.png   (열림 상태)
    ↓
RIFE (Real-Time Intermediate Flow Estimation)
    optical flow 기반 보간
    ↓
frame_001.png ~ frame_N.png
```

```python
# RIFE 사용 패턴
from rife import RIFEModel

model = RIFEModel()
frames = model.interpolate(
    img0=frame_start,
    img1=frame_end,
    n_frames=8  # 중간 프레임 수
)
```

**런타임**: ONNX Runtime  
**원본(PachiPakuGen)**: DirectML (Windows GPU)  
**우리 포팅**: ONNX Runtime + CoreML 또는 CPU fallback

### 출력 구조

```
eye/
  frame_001.png  (완전 열림)
  frame_002.png
  ...
  frame_N.png    (완전 닫힘)

mouth_a/   (あ)
  frame_001.png ~ frame_N.png
mouth_i/   (い)
mouth_u/   (う)
mouth_e/   (え)
mouth_o/   (お)
```

---

## 전체 입출력 정리

| | 입력 | 출력 |
|--|------|------|
| Stage 1 | PNG/JPG (1장) | layered PSD |
| Stage 2 | PSD + 원본 PNG | body/hair PNG (3장) |
| Stage 3 | 닫힘/열림 PNG 쌍 | frame_*.png 시퀀스 |
| 최종 | — | Live2D Cubism 소재 세트 |

---

## 성능 목표

| 항목 | 목표 |
|------|------|
| Stage 1 처리 시간 | 60초 이하 (M5 Pro MLX) |
| Stage 2 처리 시간 | 10초 이하 |
| Stage 3 처리 시간 | 5초 이하 |
| 전체 파이프라인 | 90초 이하 |

현재 See-Through 원본(A100): ~40초  
M5 Pro MLX 목표: 60초 이하
