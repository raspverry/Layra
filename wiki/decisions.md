# 설계 결정 기록 (ADR)

새로운 결정을 내릴 때마다 여기에 추가한다.  
형식: 날짜 / 결정 / 이유 / 대안 / 결과

---

## ADR-001: ML 프레임워크 → MLX 선택

**날짜**: 프로젝트 초기  
**결정**: PyTorch MPS 대신 MLX 사용  
**이유**:
- M5 Pro Neural Accelerators를 MLX만 완전 활용 가능
- Apple이 MLX를 Apple Silicon 전용으로 처음부터 설계
- Stable Diffusion 예제 이미 MLX에 존재 → SDXL 기반 See-Through 포팅 경로 있음
- 128GB UMA를 unified memory로 직접 활용 (PCIe 오버헤드 없음)

**대안**: PyTorch MPS  
**기각 이유**: MPS는 CUDA 포팅 레이어가 있어 비효율, Neural Accelerators 미활용

**결과**: LayerDiffuse + Marigold → MLX 포팅. SAM3/RIFE는 PyTorch MPS로.

---

## ADR-002: 포팅 전략 → 부분 포팅

**날짜**: 프로젝트 초기  
**결정**: 전체 MLX 포팅 대신 LayerDiffuse + Marigold만 MLX, 나머지는 MPS

**이유**:
- SAM3: PyTorch 기반, MPS에서 충분히 빠름
- RIFE: ONNX 모델 → ONNX Runtime으로 직접 실행 가능 (MLX 불필요)
- 포팅 공수 최소화, 핵심 병목(추론 시간)만 MLX로 해결

**결과**: 개발 속도 빨라짐, 성능 목표 달성 가능성 높음

---

## ADR-003: 프로덕션 GPU → RunPod A100

**날짜**: 프로젝트 초기  
**결정**: 서비스화 시 RunPod A100 Serverless 사용  
**이유**:
- 개인 서버 운영 비용 vs 필요할 때만 과금
- A100은 See-Through 원본 CUDA 코드 그대로 실행 가능
- 초기 트래픽 예측 불가 → 유연한 스케일링 필요

**대안**: AWS, GCP  
**기각 이유**: RunPod이 GPU 단위 과금에서 가격 경쟁력 있음

---

## ADR-004: Live2D → 개인 무료 플랜

**날짜**: 프로젝트 초기  
**결정**: 1차 목표(개인 아바타)는 Live2D Cubism 개인 무료 플랜 사용  
**이유**:
- 개인 비상업용 무료
- 상업화 시 Live2D 라이선스 조건 별도 검토
- Phase 3 (리깅 자동화)는 서비스 MVP에서 제외

---

## ADR-005: 서비스 MVP 범위 → Phase 1+2만

**날짜**: 프로젝트 초기  
**결정**: MVP에서 리깅 자동화(Phase 3) 제외  
**이유**:
- Live2D SDK 상업적 사용 라이선스 조건 불명확
- 리깅 자동화는 기술 난이도 매우 높음
- PSD + 애니메이션 프레임만으로도 시장 가치 있음
- 리깅은 아티스트가 직접 하는 영역, 그 앞단 자동화만으로 충분

**결과**: 빠른 출시, 리깅은 v2 검토

---

## ADR-006: 데이터 파이프라인 → See-Through 기존 weights 사용

**날짜**: 프로젝트 초기  
**결정**: 처음부터 학습하지 않고 See-Through 기존 weights 기반으로 시작  
**이유**:
- 처음부터 학습: 데이터(Live2D 모델 수천 개), 컴퓨팅(A100 수주), 팀 필요
- 1인 프로젝트 + 바이브코딩 환경에서 비현실적
- See-Through weights(Apache 2.0 코드 기반)로 시작하고 점진적 fine-tune

**주의**: weights 자체 라이선스 별도 확인 필요 (LayerDiffuse, Marigold)

---

## 추가 결정 사항

(작업하면서 새로운 결정이 생기면 여기에 추가)
