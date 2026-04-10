# Autoresearch 설정

## 개념

Karpathy의 autoresearch 방법론 적용:  
**목표 설정 → Claude Code가 루프 실행 → 아침에 결과 확인**

하나의 메트릭, 제한된 범위, 빠른 검증, 자동 rollback, git이 메모리.

참조: https://github.com/olelehmann100kMRR/autoresearch-skill

---

## 이 프로젝트에서 적용할 영역

### ✅ 적합한 작업 (수치 메트릭 있음)

| 작업 | 메트릭 | 측정 방법 |
|------|--------|----------|
| MLX 추론 속도 최적화 | 처리 시간 (초) | `time.perf_counter()` |
| 레이어 분해 품질 | IoU 점수 | 기준 출력과 비교 |
| SAM3 추출 정확도 | 마스크 IoU | 수동 라벨과 비교 |
| 배치 처리 처리량 | images/hour | 배치 실행 시간 |

### ❌ 부적합한 작업 (주관적 판단 필요)

- 아키텍처 설계
- 파이프라인 연결 로직
- UI/UX 결정

---

## 실행 패턴

### 속도 최적화 루프

```
목표: Stage 1 처리 시간 60초 이하
메트릭: 처리 시간 (낮을수록 좋음)
범위: src/stage1_layerdiff/ 의 추론 파라미터

루프:
1. 현재 처리 시간 측정 (baseline)
2. 파라미터 하나 변경 (batch size, precision, compile 등)
3. 처리 시간 재측정
4. 개선되면 keep, 아니면 revert
5. results에 기록
6. 반복
```

```bash
# Claude Code에게 전달하는 지시
"Stage 1 처리 시간을 최적화해. 
메트릭: test_images/ 의 5개 이미지 평균 처리 시간 (초)
범위: src/stage1_layerdiff/inference.py 의 파라미터만
한 번에 하나씩 변경하고, 개선되면 keep 아니면 revert
experiments/results/speed_optimization.tsv 에 기록
목표: 60초 이하"
```

### 품질 최적화 루프

```
목표: 레이어 분해 IoU 0.90 이상
메트릭: test_set 5개 이미지 평균 IoU
범위: 후처리 파라미터 (threshold, smoothing 등)

루프:
1. baseline IoU 측정
2. 파라미터 변경
3. IoU 재측정
4. 개선되면 keep, 아니면 revert
5. 기록
6. 반복
```

---

## 디렉토리 구조

```
experiments/
├── speed_optimization/
│   ├── dashboard.html      # 실시간 대시보드
│   ├── results.json        # 차트 데이터
│   ├── results.tsv         # 실험 기록
│   └── changelog.md        # 변경 이력
│
├── quality_optimization/
│   ├── dashboard.html
│   ├── results.json
│   ├── results.tsv
│   └── changelog.md
│
└── test_images/            # 기준 테스트 이미지 (5개)
    ├── test_001.png
    ├── test_002.png
    └── ...
```

---

## Claude 이미지 검수 통합

autoresearch 루프에 Claude 시각 검수를 2차 필터로 추가:

```
1차: 자동 메트릭 통과 (IoU > 0.90)
    ↓
2차: Claude 이미지 검수
    "이 두 레이어 분해 결과 중 어느 쪽이 더 자연스러워?
    A: [이미지A] vs B: [이미지B]"
    ↓
keep / discard
```

수치는 통과했지만 실제로 이상한 결과 걸러내는 용도.

---

## 결과 기록 형식

`experiments/results/[작업명].tsv`:

```tsv
experiment_id	timestamp	parameter_changed	value	metric	delta	status	notes
0	2026-04-xx	baseline	-	45.2s	-	baseline	초기 측정
1	2026-04-xx	batch_size	4	42.1s	-3.1s	keep	속도 개선
2	2026-04-xx	compile	True	55.3s	+10.1s	revert	오히려 느려짐
3	2026-04-xx	precision	float16	38.7s	-6.5s	keep	품질 유지 확인 필요
```

---

## autoresearch-log.md

실험 완료 후 요약을 여기에 기록:

```markdown
## [날짜] Stage 1 속도 최적화 Round 1

목표: 60초 이하
시작: 45.2초
결과: 38.7초 (14% 개선)
주요 발견: float16이 품질 손실 없이 속도 개선
다음 라운드: MLX compile() 옵션 재탐색
```
