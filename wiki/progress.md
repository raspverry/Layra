# 진행 상황

## 현재 단계: 🟡 Phase 0 스캐폴딩 완료, 맥북 도착 대기 중

---

## 완료된 것

- [x] 프로젝트 방향 확정 (개인 AI 아바타 → 서비스화)
- [x] 기술 스택 확정 (MLX + SAM3 + RIFE)
- [x] 위키 / 컨텍스트 시스템 구축
- [x] 파이프라인 설계 완료
- [x] 참조 레포 분석 완료 (See-Through, PachiPakuGen)
- [x] **Phase 0: 초기 코드 스캐폴딩** (2026-04-10)
  - `.gitignore`, `LICENSE` (Apache 2.0), 루트 `README.md` 추가
  - `setup.sh` 쉘 버그 수정, `requirements.txt` onnxruntime 버전 교정
  - Wiki의 SAM3 설치법 불일치 3건 수정
  - `src/common/` — 설정, 로깅, 타입, 이미지/PSD I/O 모듈
  - `src/stage1_layerdiff/` — MLX 포팅 인터페이스 (NotImplementedError)
  - `src/stage2_sam3/` — PSD 파서 + SAM3 래퍼 + 파이프라인
  - `src/stage3_rife/` — RIFE ONNX 래퍼 + 눈/입 프레임 생성
  - `src/cli.py` — typer CLI (`info`, `stage1`, `stage2`, `stage3`, `run`)
  - `experiments/results/` 시드 (speed/quality TSV + changelog)
- [x] **Phase 1a: 개발 인프라 + BLOCKER-002 해결** (2026-04-10)
  - `pyproject.toml` (editable install, `layra` CLI 엔트리포인트, ruff/mypy/pytest 설정)
  - `Makefile` (install / test / lint / format / typecheck / ci)
  - `tests/common/` pytest 스위트 — types, image_io, psd_io, config, psd_parser
  - WebFetch로 See-Through 원본 코드 정독 → BLOCKER-002 해결
  - `wiki/see-through-porting.md`에 12단계 포팅 맵 작성

---

## 맥북 도착 전 할 것

- [ ] See-Through 코드 전체 정독 (inference_psd.py 중심)
- [ ] PachiPakuGen scripts/ 디렉토리 정독
- [ ] LayerDiffuse 원본 코드 정독
- [ ] Marigold 원본 코드 정독
- [ ] Live2D Cubism 개인 플랜 가입
- [ ] RunPod 계정 생성
- [ ] RunPod에서 See-Through 원본 실행 (처리 시간, 메모리 기준 확보)
- [ ] `experiments/test_images/`에 기준 이미지 5개 추가 (본인 아바타 포함)

---

## Phase 1: MLX 포팅 (목표: 4주)

- [ ] Python 환경 세팅 (venv, MLX 설치)
- [ ] MLX SD 예제 실행 확인
- [ ] See-Through CUDA 의존성 목록 작성
- [ ] `src/stage1_layerdiff/weights.py` — PyTorch→MLX 변환 구현
- [ ] `src/stage1_layerdiff/model.py` — LayerDiffuseMLX UNet 구현
- [ ] `src/stage1_layerdiff/marigold.py` — MarigoldMLX 구현
- [ ] `src/stage1_layerdiff/inference.py` — `decompose()` 구현
- [ ] `src/common/psd_io.py::write_psd` — 플랫 저장을 진짜 레이어 PSD로 교체
- [ ] 포팅 검증 (CUDA 출력 vs MLX 출력 IoU 비교, `compute_iou` 사용)
- [ ] 처리 시간 측정 및 autoresearch로 최적화
- [ ] PSD 출력 확인

**완료 기준**: 내 일러스트 이미지 넣으면 PSD 나옴, 처리 시간 60초 이하

---

## Phase 2: SAM3 + RIFE 통합 (목표: 2주)

- [ ] SAM3 설치 및 기본 동작 확인
- [ ] 목 추출 구현 및 검증
- [ ] 입 추출 구현 및 검증
- [ ] RIFE ONNX 모델 준비 (변환 또는 다운로드)
- [ ] CoreML / CPU fallback 확인
- [ ] 눈 깜빡임 프레임 생성
- [ ] 입 5모음 프레임 생성
- [ ] 전체 파이프라인 E2E 테스트

**완료 기준**: 이미지 1장 → Live2D Cubism 소재 세트 완성

---

## Phase 3: 개인 아바타 완성 (목표: 1주)

- [ ] Live2D Cubism에 소재 임포트
- [ ] 수동 리깅 (개인용)
- [ ] 아바타 동작 테스트
- [ ] 필요시 파이프라인 개선

**완료 기준**: 실제로 움직이는 내 아바타 완성

---

## Phase 4: 서비스화 검토 (Phase 3 완료 후)

- [ ] Live2D SDK 상업 라이선스 조건 정확히 파악
- [ ] weights 라이선스 확인 (LayerDiffuse, Marigold)
- [ ] 서비스 아키텍처 설계 (Next.js + FastAPI + RunPod)
- [ ] 베타 출시 및 VTuber 커뮤니티 반응 확인

---

## 현재 블로커

`wiki/blockers.md` 참조
