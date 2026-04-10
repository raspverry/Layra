# 기술 스택

## 개발 환경

| 항목 | 스펙 |
|------|------|
| 머신 | MacBook Pro M5 Pro |
| RAM | 128GB Unified Memory |
| OS | macOS |
| Python | 3.12 |
| 개발 도구 | Claude Code |
| 로컬 LLM 서버 | vllm-mlx |

---

## ML 스택

### Stage 1: 레이어 분해

| 컴포넌트 | 기술 | 버전 | 메모 |
|---------|------|------|------|
| 프레임워크 | MLX | latest | Apple Silicon 전용 |
| 모델 베이스 | LayerDiffuse | - | SDXL 기반 |
| Depth 모델 | Marigold | - | Diffusion 기반 depth |
| 세그멘테이션 | SAM (데이터 파이프라인용) | - | 학습 데이터 생성 시 |

### Stage 2: 정밀 추출

| 컴포넌트 | 기술 | 버전 | 메모 |
|---------|------|------|------|
| 프레임워크 | PyTorch MPS | 2.x | SAM3용 |
| 세그멘테이션 | SAM3 | 3.x | Meta |
| 모델 크기 | ~3.2GB | - | 별도 다운로드 |

### Stage 3: 프레임 보간

| 컴포넌트 | 기술 | 버전 | 메모 |
|---------|------|------|------|
| 런타임 | ONNX Runtime | 2.0+ | |
| Provider | CoreML (Mac) | - | CPU fallback 있음 |
| 모델 | RIFE | - | ONNX 변환 필요 |

---

## 서비스 스택 (Phase 4 이후)

### 프론트엔드

| 항목 | 기술 | 이유 |
|------|------|------|
| 프레임워크 | Next.js | App Router, SSR |
| 언어 | TypeScript | 타입 안전성 |
| 스타일 | Tailwind CSS | 빠른 개발 |
| 상태 관리 | Zustand | 단순함 |

### 백엔드

| 항목 | 기술 | 이유 |
|------|------|------|
| API 서버 | FastAPI | Python, 비동기 |
| 작업 큐 | Celery + Redis | ML 작업 비동기 처리 |
| ML 실행 | RunPod Serverless | GPU on-demand |

### 인프라

| 항목 | 기술 | 이유 |
|------|------|------|
| DB | Supabase | PostgreSQL + Storage + Auth |
| 파일 스토리지 | Supabase Storage | PSD, PNG 저장 |
| 인증 | Clerk | 빠른 구현 |
| 결제 | Stripe | 표준 |
| GPU | RunPod A100 | 시간당 과금 |
| 배포 | Vercel (FE) + Railway (BE) | 간단함 |

---

## 주요 Python 의존성

```txt
# ML Core
mlx                      # Apple Silicon ML
torch                    # SAM3용 (MPS backend)
onnxruntime              # RIFE 실행
diffusers                # Stable Diffusion 유틸
transformers             # 모델 로딩 유틸

# Image Processing  
pillow                   # 이미지 처리
psd-tools                # PSD 파싱/생성
numpy                    # 배열 연산
opencv-python            # 이미지 처리

# SAM3
segment-anything-3       # Meta SAM3

# Utilities
tqdm                     # 진행 표시
loguru                   # 로깅
pydantic                 # 데이터 검증
```

---

## 설치 순서

```bash
# 1. venv 생성
python3.12 -m venv .venv
source .venv/bin/activate

# 2. MLX 설치
pip install mlx

# 3. PyTorch (MPS)
pip install torch torchvision

# 4. ONNX Runtime
pip install onnxruntime

# 5. 나머지
pip install diffusers transformers pillow psd-tools numpy opencv-python
pip install segment-anything-3
pip install tqdm loguru pydantic

# 6. 모델 weights 다운로드
# See-Through: scripts/download_models.py 참조
# SAM3: ~3.2GB
```

---

## 환경 변수

```env
# .env (git에 올리지 말 것)

# RunPod (서비스화 시)
RUNPOD_API_KEY=

# Supabase (서비스화 시)
SUPABASE_URL=
SUPABASE_ANON_KEY=

# 모델 경로
MODEL_DIR=/path/to/models
SAM3_MODEL_PATH=/path/to/models/sam3_vit_h.pth
RIFE_MODEL_PATH=/path/to/models/rife.onnx
```

---

## 참고 링크

- [MLX 공식 문서](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX Examples (SD 포함)](https://github.com/ml-explore/mlx-examples)
- [See-Through 레포](https://github.com/shitagaki-lab/see-through)
- [PachiPakuGen 레포](https://github.com/kazuya-bros/PachiPakuGen)
- [LayerDiffuse 레포](https://github.com/lllyasviel/LayerDiffuse)
- [Marigold 레포](https://github.com/prs-eth/marigold)
- [ONNX Runtime Mac](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
