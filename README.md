# Layra

> 애니 일러스트 1장 → MLX 기반 레이어 분해 → SAM3/RIFE 애니메이션 프레임 생성 → Live2D 아바타 소재

Apple Silicon(M5 Pro, 128GB UMA) 환경에서 단일 일러스트로부터 Live2D Cubism 호환 소재 세트를 자동 생성하는 파이프라인입니다.

## 파이프라인

```
[입력] PNG/JPG 1장
   ↓
[Stage 1] LayerDiffuse + Marigold (MLX 포팅)
          최대 23개 RGBA 레이어로 분해 → PSD
   ↓
[Stage 2] SAM3 (PyTorch MPS)
          목/입 영역 정밀 추출
   ↓
[Stage 3] RIFE (ONNX Runtime / CoreML)
          눈 깜빡임 + 5모음 프레임 생성
   ↓
[출력] Live2D Cubism 호환 소재 세트
```

성능 목표: M5 Pro에서 전체 90초 이하 (Stage1 60 + Stage2 10 + Stage3 5).

## 디렉토리

| 경로 | 내용 |
|------|------|
| `CLAUDE.md` | Claude Code 진입점 / 작업 규칙 |
| `wiki/` | 프로젝트 문서 (설계, 결정, 진행 상황) |
| `src/common/` | 공통 유틸 (설정, 로깅, 이미지/PSD I/O) |
| `src/stage1_layerdiff/` | LayerDiffuse + Marigold MLX 포팅 |
| `src/stage2_sam3/` | SAM3 목/입 추출 |
| `src/stage3_rife/` | RIFE 프레임 보간 |
| `src/cli.py` | `layra` CLI 엔트리포인트 |
| `experiments/` | autoresearch 실험 결과 |

## 문서

세션 시작 시 아래 순서로 읽는 것을 권장합니다:

1. [`wiki/overview.md`](wiki/overview.md) — 프로젝트 전체 구조
2. [`wiki/pipeline.md`](wiki/pipeline.md) — 기술 파이프라인 상세
3. [`wiki/progress.md`](wiki/progress.md) — 현재 진행 상황
4. [`wiki/blockers.md`](wiki/blockers.md) — 미해결 이슈
5. [`wiki/decisions.md`](wiki/decisions.md) — 설계 결정 기록 (ADR)

## 빠른 시작 (macOS Apple Silicon)

```bash
git clone https://github.com/raspverry/layra
cd layra
chmod +x setup.sh
./setup.sh
source .venv/bin/activate

# Stage 1 단독 실행
python -m src.cli stage1 path/to/image.png --output out.psd

# 전체 파이프라인
python -m src.cli run path/to/image.png --output-dir out/
```

자세한 환경 세팅은 [`wiki/day1-checklist.md`](wiki/day1-checklist.md) 참조.

## 기술 스택

- **ML 프레임워크**: MLX (Stage 1), PyTorch MPS (Stage 2), ONNX Runtime (Stage 3)
- **언어**: Python 3.12
- **CLI**: typer + rich + loguru
- **이미지**: Pillow, psd-tools, OpenCV, NumPy

## 라이선스

Apache License 2.0. 자세한 내용은 [`LICENSE`](LICENSE) 참조.

참조 모델의 weights 라이선스는 별도 확인 필요:
- [LayerDiffuse](https://github.com/lllyasviel/LayerDiffuse)
- [Marigold](https://github.com/prs-eth/marigold)
- [Segment Anything 3](https://github.com/facebookresearch/segment-anything-3)
- [RIFE](https://github.com/hzwer/RIFE)

## 참조

- [See-Through (Stage 1 베이스)](https://github.com/shitagaki-lab/see-through) — Apache 2.0
- [PachiPakuGen (Stage 2+3 로직 참조)](https://github.com/kazuya-bros/PachiPakuGen) — MIT
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
