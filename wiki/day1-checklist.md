# Day 1 체크리스트 (맥북 도착 당일)

## 순서대로 진행

### 1. 환경 세팅 (30분)

```bash
# 레포 클론
git clone https://github.com/your-username/see-through-project
cd see-through-project

# 환경 세팅 스크립트 실행
chmod +x setup.sh
./setup.sh
```

- [ ] Python 3.12 설치 확인
- [ ] MLX 설치 및 동작 확인
- [ ] PyTorch MPS 사용 가능 확인
- [ ] ONNX Runtime CoreML provider 확인

---

### 2. MLX Stable Diffusion 예제 실행 (1시간)

```bash
git clone https://github.com/ml-explore/mlx-examples
cd mlx-examples/stable_diffusion
pip install -r requirements.txt
python txt2image.py --prompt "anime girl portrait" --n_images 1
```

- [ ] SD 예제 실행 성공
- [ ] 처리 시간 기록 (___초)
- [ ] VRAM 사용량 확인 (Activity Monitor)

**이 단계가 성공하면 See-Through MLX 포팅 경로 확인됨.**

---

### 3. See-Through 원본 코드 분석 (2시간)

```bash
git clone https://github.com/shitagaki-lab/see-through
cd see-through
```

확인할 것:
```bash
# CUDA 의존성 목록
grep -rn "cuda\|xformers\|triton\|flash_attn" src/ --include="*.py"

# 모델 아키텍처
cat src/layerdiff/models/*.py

# 추론 스크립트 전체
cat inference/scripts/inference_psd.py
```

- [ ] LayerDiffuse 아키텍처 파악 (표준 UNet인지, 커스텀 ops 있는지)
- [ ] Marigold 아키텍처 파악
- [ ] CUDA 전용 코드 목록 작성 → `wiki/see-through-porting.md` 업데이트
- [ ] BLOCKER-002 해결

---

### 4. PachiPakuGen 코드 분석 (1시간)

```bash
git clone https://github.com/kazuya-bros/PachiPakuGen
cd PachiPakuGen
cat scripts/*.py
```

- [ ] SAM3 사용 방식 파악
- [ ] RIFE ONNX 모델 구조 파악
- [ ] PSD 레이어 파싱 방식 파악
- [ ] `wiki/sam3-rife.md` 업데이트

---

### 5. RunPod에서 See-Through 원본 실행 (기준 측정)

```bash
# RunPod A100 인스턴스 생성
# Docker: nvidia/cuda:12.8.0-runtime-ubuntu22.04

# See-Through 설치 및 실행
git clone https://github.com/shitagaki-lab/see-through
cd see-through
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 기준 이미지로 실행 및 시간 측정
time python inference/scripts/inference_psd.py \
    --srcp assets/test_image.png \
    --save_to_psd
```

- [ ] 처리 시간 기록 (___초) → MLX 목표 설정
- [ ] 출력 PSD 저장 → MLX 포팅 검증 기준으로 사용
- [ ] BLOCKER-004 해결

---

### 6. Day 1 마무리

- [ ] `wiki/progress.md` 업데이트
- [ ] `wiki/blockers.md` 업데이트
- [ ] 내일 할 것 메모

---

## Day 2 이후

`wiki/progress.md` Phase 1 체크리스트 따라 진행.

MLX 포팅 시작: `wiki/see-through-porting.md` Step 1부터.
