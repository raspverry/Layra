# See-Through MLX 포팅

## 상태: 🟢 템플릿 완료, 맥북 도착 시 MLX 연산만 채우면 됨

- **Phase 1a (2026-04-10)**: 원본 코드 분석 → 포팅 가능성 확정
- **Phase 1b (2026-04-10)**: MLX SD 예제 분석 → 12단계 포팅 맵
- **Phase 1c (2026-04-10)**: Stage 1 전체 템플릿 프리뷰 작성
  - `src/stage1_layerdiff/configs.py`: 모든 dataclass config 확정
  - `src/stage1_layerdiff/mlx_ops/`: 프리미티브 6개 (embeddings, norms,
    attention, resnet, blocks, Transformer3DModel/CrossFrame)
  - `src/stage1_layerdiff/schedulers.py`: **DPM++ 2M SDE 수학적으로 완전 구현**
    (numpy 기반, Karras sigmas 포함, 20개 unit test)
  - `src/stage1_layerdiff/weights.py`: PyTorch safetensors → MLX 변환 완전 스캐폴딩
  - `src/stage1_layerdiff/unet_frame.py`, `vae.py`, `model.py`, `marigold.py`:
    클래스 구조와 forward 의사코드가 docstring에 상세히 기록됨
  - CI: 74/74 pytest + 0 mypy errors (33 source files)

**맥북 도착 후 할 일**: 각 모듈의 `NotImplementedError` 위치에서
주석으로 적어둔 MLX 연산만 구현하면 된다. 인터페이스, config, 수학 로직은
이미 확정된 상태.

---

## 원본 코드 분석 결과 (2026-04-10)

WebFetch로 shitagaki-lab/see-through@main을 분석한 결과:

### requirements.txt 수준

`torch==2.8.0+cu128`은 핀닝되어 있지만, CUDA 최적화 의존성은
**전혀 없다**:

| 패턴 | 결과 |
|------|------|
| `xformers` | ❌ 없음 |
| `triton` | ❌ 없음 |
| `flash_attn` / `flash-attention` | ❌ 없음 |
| `bitsandbytes` | ❌ 기본 없음 (NF4 quant는 `requirements-inference-bnb.txt`로 분리) |

즉 메모리 효율 attention이나 커스텀 CUDA 커널 의존성이 없다.

### inference_psd.py (메인 추론 스크립트)

```python
# 실제 임포트 (전체)
import torch
from utils.io_utils import find_all_imgs
from utils import inference_utils
from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
from utils.torch_utils import seed_everything
```

CUDA 특화 코드 검색 결과:
- `cuda`, `xformers`, `flash_attn`, `triton`, `torch.compile`,
  `scaled_dot_product_attention`, `enable_xformers`, `bitsandbytes`, `NF4`
  **모두 0건**
- 유일한 GPU 특화는 `--group_offload` 플래그뿐 (128GB UMA에서는 불필요)

### common/utils/inference_utils.py

`apply_layerdiff`, `apply_marigold`의 실제 모델 로드/추론 로직:

```python
# LayerDiffuse 로딩
unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet')
pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
    pretrained, trans_vae=trans_vae, unet=unet, scheduler=None,
)

# 디바이스 배치 — 전부 표준 .to() 패턴
vae.to(dtype=torch.bfloat16, device='cuda')
trans_vae.to(dtype=torch.bfloat16, device='cuda')
unet.to(dtype=torch.bfloat16, device='cuda')
text_encoder.to(dtype=torch.bfloat16, device='cuda')
text_encoder_2.to(dtype=torch.bfloat16, device='cuda')
```

- `xformers`, `flash_attn`, `triton`, `torch.compile`, `sdpa`,
  `memory_efficient_attention` **전부 0건**
- 유일한 CUDA 의존 코드는 `.to(device='cuda')`와
  `enable_group_offload('cuda', ...)`뿐 → MLX에서는 그냥 제거

허깅페이스 허브 ID:
- LayerDiffuse: `layerdifforg/seethroughv0.0.2_layerdiff3d`
- Marigold:    `24yearsold/seethroughv0.0.1_marigold`

### common/modules/layerdiffuse/ 구조

```
layerdiffuse/
├── __init__.py
├── diffusers_kdiffusion_sdxl.py    # KDiffusionStableDiffusionXLPipeline
├── layerdiff3d.py                  # UNetFrameConditionModel
├── transformer3d.py                # Transformer3DModel, CrossFrameTransformerBlock
├── vae.py                          # TransparentVAE
└── utils.py
```

**KDiffusionStableDiffusionXLPipeline**:
```python
class KDiffusionStableDiffusionXLPipeline(StableDiffusionXLImg2ImgPipeline):
```
→ 표준 diffusers `StableDiffusionXLImg2ImgPipeline` 서브클래스.
  스케줄러는 DPM++ 계열 (`DPMPP_2M_SDE` 기본, Karras sigmas 지원).
  `enable_xformers_memory_efficient_attention()` 호출 **없음**.

**transformer3d.py** (핵심):
```python
from diffusers.models.attention import (
    BasicTransformerBlock, FeedForward,
    _chunked_feed_forward, TemporalBasicTransformerBlock,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    ImagePositionalEmbeddings, PatchEmbed, PixArtAlphaTextProjection,
)
from diffusers.models.normalization import AdaLayerNormSingle
```

- 정의 클래스: `CrossFrameTransformerBlock`, `Transformer3DModel`
- 커스텀 attention 없음 — diffusers의 표준 `Attention` 클래스 사용
- `@torch.compile` 데코레이터 없음
- 3D 텐서 구조: `[batch_size, seq_length, num_frames, channels]`로
  reshape — `num_frames`가 23개 레이어 차원

### common/modules/marigold/ 구조

```
marigold/
├── __init__.py
├── marigold_depth_pipeline.py    # MarigoldDepthPipeline
└── multi_res_noise.py            # multi-res noise scheduling
```

표준 prs-eth/marigold의 See-Through fine-tune 버전.
Stable Diffusion 1.5 기반 단안 depth diffusion → MLX SD 예제에서
가장 쉽게 포팅할 수 있는 컴포넌트.

---

## 포팅 맵 (MLX 컴포넌트별)

이 섹션이 실제 포팅 작업의 우선순위 리스트다.

| # | 원본 (PyTorch / diffusers) | MLX 목표 | 난이도 | 비고 |
|---|---------------------------|---------|--------|------|
| 1 | `diffusers.models.attention_processor.Attention` | `src/stage1_layerdiff/mlx_ops/attention.py` | ⭐⭐ | MLX SD 예제에 근사 구현 있음 |
| 2 | `diffusers.models.attention.BasicTransformerBlock` | `src/stage1_layerdiff/mlx_ops/blocks.py` | ⭐⭐ | self-attn + cross-attn + FF |
| 3 | `diffusers.models.attention.TemporalBasicTransformerBlock` | 위 파일 | ⭐⭐⭐ | 시간 축 처리 추가 |
| 4 | `diffusers.models.normalization.AdaLayerNormSingle` | `src/stage1_layerdiff/mlx_ops/norms.py` | ⭐ | 간단 |
| 5 | `CrossFrameTransformerBlock` (layerdiffuse 커스텀) | `src/stage1_layerdiff/transformer3d.py` | ⭐⭐⭐ | 레이어 간 cross-attn |
| 6 | `Transformer3DModel` | 위 파일 | ⭐⭐⭐ | 3D reshape 로직 |
| 7 | `UNetFrameConditionModel` (layerdiff3d.py) | `src/stage1_layerdiff/model.py` | ⭐⭐⭐⭐ | SDXL UNet + frame cond |
| 8 | `TransparentVAE` (vae.py) | `src/stage1_layerdiff/vae.py` | ⭐⭐⭐ | 표준 VAE + 알파 채널 |
| 9 | DPM++ 2M SDE 스케줄러 | `src/stage1_layerdiff/schedulers.py` | ⭐⭐ | 수치 알고리즘, CPU/MLX 모두 가능 |
| 10 | `KDiffusionStableDiffusionXLPipeline` | `src/stage1_layerdiff/pipeline.py` | ⭐⭐⭐ | 오케스트레이션 |
| 11 | `MarigoldDepthPipeline` | `src/stage1_layerdiff/marigold.py` | ⭐⭐ | 표준 SD1.5 depth |
| 12 | `weights.py` — PyTorch state_dict → MLX | 이미 스켈레톤 있음 | ⭐⭐ | key mapping 필요 |

### Layra 측 포팅 템플릿 위치 (Phase 1c 결과)

| 원본 / MLX SD 대응 | Layra 파일 | 상태 |
|---|---|---|
| UNetConfig, AutoencoderConfig, DiffusionConfig | `configs.py` | ✅ 값 확정 |
| TimestepEmbedding, SDXL addition embedding | `mlx_ops/embeddings.py` | 🟡 스켈레톤 |
| AdaLayerNormSingle | `mlx_ops/norms.py` | 🟡 스켈레톤 |
| TransformerBlock + CrossFrameTransformerBlock | `mlx_ops/attention.py` | 🟡 스켈레톤 |
| ResnetBlock2D | `mlx_ops/resnet.py` | 🟡 스켈레톤 |
| Transformer3DModel + UNetBlock2D | `mlx_ops/blocks.py` | 🟡 스켈레톤 |
| UNetFrameConditionModel (SDXL+frame) | `unet_frame.py` | 🟡 스켈레톤 (forward 의사코드) |
| TransparentVAE | `vae.py` | 🟡 스켈레톤 |
| DPMSolverMultistep (DPM++ 2M SDE) | `schedulers.py` | ✅ **numpy 완전 구현 + 20 tests** |
| PyTorch→MLX weights | `weights.py` | ✅ KEY_MAPS + convert_state_dict 완료 |
| LayerDiffuseMLX 고수준 | `model.py` | 🟡 sample() 의사코드 |
| MarigoldMLX | `marigold.py` | 🟡 predict_depth() 의사코드 |

### MLX SD 예제에서 재사용 가능한 부분 (2026-04-10 분석)

https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion 의
`stable_diffusion/` 패키지를 정독한 결과:

```
stable_diffusion/
├── unet.py         # UNetModel + UNetBlock2D + Transformer2D + ResnetBlock2D
├── vae.py          # Autoencoder
├── clip.py         # Text encoder (CLIP)
├── tokenizer.py    # BPE tokenizer
├── sampler.py      # Diffusion sampler
├── model_io.py     # Weight loading utilities
└── config.py       # Dataclass configs
```

**MLX UNet 클래스 구조** (unet.py):

| MLX 클래스 | 역할 | diffusers 대응 |
|-----------|------|----------------|
| `UNetModel(nn.Module)` | UNet 본체 | `UNet2DConditionModel` |
| `UNetBlock2D(nn.Module)` | down/up 블록 | `CrossAttnDownBlock2D`/`CrossAttnUpBlock2D` |
| `ResnetBlock2D(nn.Module)` | residual | `ResnetBlock2D` |
| `Transformer2D(nn.Module)` | attention 그룹 | `Transformer2DModel` |
| `TransformerBlock(nn.Module)` | 단일 TF 블록 | `BasicTransformerBlock` |
| `TimestepEmbedding(nn.Module)` | time 임베딩 | `Timesteps + TimestepEmbedding` |

**`UNetModel.__call__` 시그니처**:
```python
def __call__(self, x, timestep, encoder_x,
             attn_mask=None, encoder_attn_mask=None, text_time=None):
```
- `x`: latent `(B, C, H, W)`
- `timestep`: diffusion step
- `encoder_x`: text embedding
- `text_time`: pooled embedding (SDXL micro-conditioning)

**사용 레이어** (전부 `mlx.nn`):
`Conv2d`, `GroupNorm`, `LayerNorm`, `Linear`,
`SinusoidalPositionalEncoding`, `MultiHeadAttention`
(cross-attn은 `key_input_dims` 분리).

**LayerDiffuse 포팅 전략**:
1. `UNetModel`을 베이스로 복사 → `UNetFrameConditionModel` (추가 차원 `num_frames=23`)
2. `Transformer2D` → `Transformer3D` (cross-frame attention 추가)
3. `TransformerBlock` → 그대로 사용, `CrossFrameTransformerBlock` 추가
4. `TimestepEmbedding`, `ResnetBlock2D`, `UNetBlock2D` — 거의 그대로 사용
5. `model_io.py`를 참조하여 `weights.py` 작성 (key 매핑만 추가)
6. `sampler.py`를 참조하여 k-diffusion DPM++ 2M SDE 구현
7. Marigold: 표준 SD 1.5 기반이라 `UNetModel` 축소 버전으로 충분

**Marigold 구조** (prs-eth/marigold + See-Through fine-tune):
```
common/modules/marigold/
├── marigold_depth_pipeline.py
└── multi_res_noise.py
```
→ MLX SD 예제의 `sampler.py` + `UNetModel` 소수만 수정해서 포팅 가능.

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
| 2026-04-10 | `src/stage1_layerdiff/` 스캐폴딩 | ✅ | `Stage1Pipeline`, `LayerDiffuseMLX`, `MarigoldMLX`, `weights.py` 인터페이스 정의. 전부 NotImplementedError. |
| 2026-04-10 | `src/common/psd_io.py` `parse_psd` 구현 | ✅ | psd-tools 기반. `LayerName` StrEnum 매칭. |
| 2026-04-10 | `src/common/psd_io.py` `write_psd` 임시 구현 | ⚠️ | Pillow로 플랫 PSD 저장. Stage 1 완성 시 진짜 레이어 PSD 작성기로 교체 필요. |
| 2026-04-10 | See-Through 원본 코드 분석 (WebFetch) | ✅ | BLOCKER-002 해결. xformers/triton/flash_attn/compile 전부 없음 확인. 포팅 맵 작성. |

---

## 참고 자료

- [MLX Stable Diffusion 예제](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion)
- [MLX 공식 문서](https://ml-explore.github.io/mlx/)
- [See-Through 원본 코드](https://github.com/shitagaki-lab/see-through)
- [LayerDiffuse 원본](https://github.com/lllyasviel/LayerDiffuse)
- [Marigold 원본](https://github.com/prs-eth/marigold)
