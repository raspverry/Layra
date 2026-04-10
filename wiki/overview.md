# 프로젝트 개요

## 목표

애니메이션 일러스트 이미지 1장을 입력으로 받아:
1. 의미론적 레이어로 분해 (최대 23개)
2. 눈/입 애니메이션 프레임 자동 생성
3. Live2D 호환 아바타 소재 출력

**1차 목표**: 개인 AI VTuber 아바타 제작  
**2차 목표**: 서비스화 검토 (반응 보고 결정)

---

## 전체 파이프라인

```
[입력] 애니 일러스트 PNG/JPG
         ↓
[Stage 1] See-Through (MLX 포팅)
         레이어 분해 → 최대 23개 RGBA 레이어
         Depth 추정 → drawing order 결정
         출력: layered PSD
         ↓
[Stage 2] SAM3 + PachiPakuGen 로직
         목/입 영역 정밀 추출
         See-Through 단독으로는 경계선 부자연스러움
         출력: 정제된 body/hair/hair_back PNG
         ↓
[Stage 3] RIFE 프레임 보간
         눈 깜빡임 프레임 생성
         입 모양 5모음 프레임 생성 (あいうえお)
         출력: frame_001.png ~ frame_N.png
         ↓
[출력] Live2D Cubism 호환 소재
       body.png / hair.png / hair_back.png
       eye/frame_*.png
       mouth_a~/frame_*.png
```

---

## 참조 레포지토리

| 레포 | 역할 | 라이선스 |
|------|------|---------|
| [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through) | Stage 1 베이스 | Apache 2.0 |
| [kazuya-bros/PachiPakuGen](https://github.com/kazuya-bros/PachiPakuGen) | Stage 2+3 로직 참조 | MIT |
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | Apple Silicon ML 프레임워크 | MIT |
| [lllyasviel/LayerDiffuse](https://github.com/lllyasviel/LayerDiffuse) | See-Through 내부 모델 | Apache 2.0 |
| [prs-eth/marigold](https://github.com/prs-eth/marigold) | Depth estimation | Apache 2.0 |

---

## 논문

- **See-Through**: "Single-image Layer Decomposition for Anime Characters"  
  SIGGRAPH 2026 Conditionally Accepted  
  arXiv: 2602.03749

---

## 핵심 기술 결정

| 결정 | 선택 | 이유 |
|------|------|------|
| ML 프레임워크 | MLX | Apple Silicon 전용, 128GB UMA 활용 |
| 포팅 전략 | LayerDiffuse+Marigold만 MLX, SAM3/RIFE는 PyTorch MPS | 우선순위 |
| 프로덕션 GPU | RunPod A100 | 서비스화 시점에 |
| Live2D | Cubism 개인 무료 플랜 | 1차 목표가 개인용 |

자세한 결정 이유 → `decisions.md`

---

## 디렉토리 구조

```
see-through-project/
├── CLAUDE.md               # Claude Code 진입점
├── wiki/                   # 프로젝트 위키
│   ├── overview.md         # 이 파일
│   ├── pipeline.md         # 파이프라인 상세
│   ├── see-through-porting.md  # MLX 포팅 진행
│   ├── sam3-rife.md        # SAM3+RIFE 통합
│   ├── decisions.md        # 설계 결정 기록
│   ├── blockers.md         # 미해결 이슈
│   ├── progress.md         # 진행 상황
│   ├── autoresearch.md     # autoresearch 설정
│   └── autoresearch-log.md # 실험 기록
├── src/                    # 프로덕션 코드
│   ├── stage1_layerdiff/   # See-Through MLX 포팅
│   ├── stage2_sam3/        # SAM3 추출
│   └── stage3_rife/        # RIFE 프레임 보간
└── experiments/            # 실험 코드
    └── results/            # 실험 결과
```
