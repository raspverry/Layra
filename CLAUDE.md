# CLAUDE.md — See-Through Project

## 세션 시작 시 반드시 먼저 읽을 것

```
wiki/overview.md        → 프로젝트 전체 구조
wiki/pipeline.md        → 기술 파이프라인 상세
wiki/progress.md        → 현재 진행 상황 + 다음 할 일
wiki/blockers.md        → 막힌 것들 / 미해결 이슈
```

## 작업 완료 후 반드시 업데이트

- 새로운 결정 내렸으면 → `wiki/decisions.md`
- 포팅 진행됐으면 → `wiki/see-through-porting.md`
- 막힌 거 해결됐으면 → `wiki/blockers.md`
- 진행 상황 바뀌었으면 → `wiki/progress.md`

위키는 항상 최신 상태를 유지한다. 작업 끝날 때마다 업데이트 필수.

## 프로젝트 한 줄 요약

애니메이션 일러스트 1장 → MLX 기반 레이어 분해 → SAM3/RIFE 애니메이션 프레임 생성 → Live2D 아바타

## 개발 환경

- **머신**: MacBook Pro M5 Pro 128GB Unified Memory
- **OS**: macOS
- **ML 프레임워크**: MLX (Apple Silicon 전용)
- **개발 도구**: Claude Code + vllm-mlx (로컬 LLM)
- **언어**: Python (ML), TypeScript (서비스 레이어)

## 코드 컨벤션

- Python: 타입 힌트 필수, docstring 필수
- 실험 코드는 `experiments/` 디렉토리에
- 프로덕션 코드는 `src/` 디렉토리에
- 모든 ML 실험은 `experiments/results/`에 결과 저장

## autoresearch 루프

수치 최적화가 필요한 작업은 autoresearch 패턴 적용:
1. 메트릭 정의 (처리 시간, IoU 등)
2. 파라미터 변경
3. 측정
4. keep / discard
5. `wiki/autoresearch-log.md`에 기록

자세한 내용 → `wiki/autoresearch.md`
