# Stage 3: RIFE 프레임 보간

## 현재 구조 (스캐폴딩)

```
stage3_rife/
├── __init__.py         # 공개 API
├── pipeline.py         # Stage3Pipeline, Stage3Inputs
├── interpolator.py     # RIFEInterpolator (ONNX Runtime)
├── eye_blink.py        # generate_eye_blink_frames
└── mouth_frames.py     # generate_mouth_frames (VOWELS = a i u e o)
```

## 상태

- `interpolator`: 로드는 lazy, CoreML → CPU fallback 자동.
- `eye_blink` / `mouth_frames`: 실제 호출 가능 (RIFE 모델만 있으면 동작).
- RIFE ONNX 모델은 변형이 많아 입력 키(`timestep` 유무)는 런타임에 확인.

## 시작 전 읽을 것

1. `wiki/sam3-rife.md` (RIFE 섹션)
2. PachiPakuGen RIFE 통합 코드 참조
3. ONNX Runtime CoreML Provider 문서
