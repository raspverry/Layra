# See-Through MLX 포팅

## 상태: 🔴 미시작 (맥북 도착 후 시작)

---

## 포팅 범위

| 컴포넌트 | 포팅 방법 | 우선순위 |
|---------|----------|---------|
| LayerDiffuse (SDXL UNet) | CUDA → MLX | 🔴 필수 |
| Marigold (depth diffusion) | CUDA → MLX | 🔴 필수 |
| SAM3 | CUDA → PyTorch MPS | 🟡 2순위 |
| RIFE | DirectML → ONNX CPU/CoreML | 🟡 2순위 |

---

## 포팅 전 체크리스트

### 환경 준비
- [ ] Python 3.12 venv 생성
- [ ] MLX 설치: `pip install mlx`
- [ ] MLX 예제 실행 확인: `mlx-examples/stable_diffusion`
- [ ] See-Through 레포 클론
- [ ] 원본 의존성 파악: `pip install -r requirements.txt`
- [ ] 모델 weights 다운로드

### 의존성 파악
원본 See-Through의 CUDA 의존성:
```
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
diffusers (CUDA 최적화 포함)
xformers (CUDA 전용)
```

MLX 대체:
```
mlx
mlx-examples (SD 참조용)
diffusers (MPS 백엔드로 fallback 가능)
```

---

## 포팅 단계

### Step 1: 모델 아키텍처 파악

```bash
# See-Through 코드에서 모델 구조 파악
cat inference/scripts/inference_psd.py
cat src/layerdiff/models/  # LayerDiff 모델 정의
cat src/marigold/          # Marigold 모델 정의
```

파악해야 할 것:
- UNet 구조 (표준 SDXL UNet인지, 커스텀인지)
- 커스텀 CUDA 커널 사용 여부
- xformers attention 사용 여부

### Step 2: CUDA 의존성 제거

```python
# 제거 대상 패턴들
import xformers  # → 제거, MLX attention 사용
model.cuda()     # → 제거 (MLX는 unified memory)
tensor.to("cuda") # → 제거
torch.cuda.amp.autocast()  # → mlx 자체 precision 관리
```

### Step 3: MLX 모델 재구현

```python
# PyTorch UNet → MLX UNet 변환 패턴
# PyTorch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(4, 320, 3, padding=1)
    def forward(self, x):
        return self.conv(x)

# MLX
import mlx.nn as nn
import mlx.core as mx
class UNet(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(4, 320, 3, padding=1)
    def __call__(self, x):
        return self.conv(x)
```

### Step 4: Weights 로드

```python
# PyTorch weights → MLX 변환
import torch
import mlx.core as mx
import numpy as np

def convert_weights(pytorch_path: str, mlx_path: str):
    state_dict = torch.load(pytorch_path, map_location="cpu")
    mlx_weights = {}
    for key, tensor in state_dict.items():
        mlx_weights[key] = mx.array(tensor.numpy())
    mx.savez(mlx_path, **mlx_weights)
```

### Step 5: 추론 파이프라인 재구현

```python
# diffusion sampling loop (MLX)
import mlx.core as mx

def ddim_sample(model, latents, timesteps, ...):
    for t in timesteps:
        noise_pred = model(latents, t, ...)
        latents = scheduler_step(noise_pred, t, latents)
        mx.eval(latents)  # lazy evaluation 강제 실행
    return latents
```

**중요**: MLX는 lazy evaluation → `mx.eval()` 명시적 호출 필요

### Step 6: 검증

```python
# 포팅 검증: CUDA 출력 vs MLX 출력 비교
def validate_porting(image_path: str):
    # 동일 입력에 대해
    # CUDA 결과 (RunPod에서 미리 저장해둔 것)
    cuda_output = load_reference("reference_cuda_output.npz")
    
    # MLX 결과
    mlx_output = run_mlx_pipeline(image_path)
    
    # 레이어별 IoU 비교
    for i, (cuda_layer, mlx_layer) in enumerate(zip(cuda_output, mlx_output)):
        iou = compute_iou(cuda_layer, mlx_layer)
        print(f"Layer {i}: IoU = {iou:.4f}")
        assert iou > 0.90, f"Layer {i} IoU too low: {iou}"
```

---

## 알려진 문제 / 주의사항

### xformers
원본이 xformers memory-efficient attention 쓸 경우:
- MLX에는 자체 efficient attention 있음
- `mlx.nn.MultiHeadAttention` 사용

### group_offload
원본의 group offload 기능:
- MLX는 unified memory라 offload 개념 없음
- 128GB RAM이면 전체 모델 올려도 됨
- 이 옵션 무시해도 됨

### bf16
원본: `bf16 precision`
MLX: `mx.bfloat16` 지원 → 그대로 사용 가능

### SDXL VAE
latent space 인코딩/디코딩:
- MLX SD 예제에 VAE 구현 있음 → 참조

---

## 진행 상황 로그

| 날짜 | 작업 | 상태 | 메모 |
|------|------|------|------|
| (미시작) | | | |

---

## 참고 자료

- [MLX Stable Diffusion 예제](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion)
- [MLX 공식 문서](https://ml-explore.github.io/mlx/)
- [See-Through 원본 코드](https://github.com/shitagaki-lab/see-through)
- [LayerDiffuse 원본](https://github.com/lllyasviel/LayerDiffuse)
- [Marigold 원본](https://github.com/prs-eth/marigold)
