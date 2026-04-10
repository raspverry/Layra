#!/bin/bash
# ============================================================
# See-Through Project — Mac M5 Pro 환경 세팅 스크립트
# ============================================================

set -e  # 에러 시 중단

echo "=== See-Through Project 환경 세팅 ==="
echo ""

# Python 버전 확인
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" < "3.12" ]]; then
    echo "⚠️  Python 3.12 이상 필요. brew install python@3.12"
    exit 1
fi

# venv 생성
echo ""
echo "--- venv 생성 ---"
python3.12 -m venv .venv
source .venv/bin/activate
echo "✅ venv 활성화: .venv"

# pip 업그레이드
pip install --upgrade pip --quiet

# MLX 설치
echo ""
echo "--- MLX 설치 ---"
pip install mlx --quiet
python -c "import mlx.core as mx; print(f'✅ MLX 설치 완료: {mx.__version__}')"

# PyTorch MPS
echo ""
echo "--- PyTorch (MPS) 설치 ---"
pip install torch torchvision --quiet
python -c "
import torch
mps = torch.backends.mps.is_available()
print(f'✅ PyTorch 설치 완료: {torch.__version__}')
print(f'   MPS 사용 가능: {mps}')
"

# ONNX Runtime
echo ""
echo "--- ONNX Runtime 설치 ---"
pip install onnxruntime --quiet
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'✅ ONNX Runtime 설치 완료')
print(f'   사용 가능한 providers: {providers}')
"

# 나머지 의존성
echo ""
echo "--- 나머지 의존성 설치 ---"
pip install diffusers transformers accelerate safetensors --quiet
pip install Pillow psd-tools numpy opencv-python --quiet
pip install tqdm loguru pydantic python-dotenv typer rich --quiet
echo "✅ 의존성 설치 완료"

# SAM3 (별도)
echo ""
echo "--- SAM3 설치 ---"
if ! pip install git+https://github.com/facebookresearch/segment-anything-3.git --quiet 2>/dev/null; then
    echo "⚠️  SAM3 설치 실패. 수동으로 설치 필요:"
    echo "   pip install git+https://github.com/facebookresearch/segment-anything-3.git"
fi

# 디렉토리 구조 생성
echo ""
echo "--- 디렉토리 구조 생성 ---"
mkdir -p models
mkdir -p experiments/test_images
mkdir -p experiments/results/speed_optimization
mkdir -p experiments/results/quality_optimization
echo "✅ 디렉토리 생성 완료"

# MLX Stable Diffusion 예제 확인
echo ""
echo "--- MLX SD 예제 확인 ---"
echo "MLX Stable Diffusion 예제를 확인하려면:"
echo "  git clone https://github.com/ml-explore/mlx-examples"
echo "  cd mlx-examples/stable_diffusion"
echo "  python txt2image.py --prompt 'test'"

# 완료
echo ""
echo "============================================"
echo "✅ 환경 세팅 완료"
echo ""
echo "다음 단계:"
echo "  1. source .venv/bin/activate"
echo "  2. See-Through 레포 클론:"
echo "     git clone https://github.com/shitagaki-lab/see-through"
echo "  3. PachiPakuGen 레포 클론:"  
echo "     git clone https://github.com/kazuya-bros/PachiPakuGen"
echo "  4. MLX SD 예제 실행 확인"
echo "  5. wiki/see-through-porting.md 의 포팅 시작"
echo "============================================"
