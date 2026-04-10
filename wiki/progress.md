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
- [x] **Phase 1b: Claude Code 세션 하네스 + 원본 분석 확장** (2026-04-10)
  - `.claude/hooks/session-start.sh` + `.claude/settings.json` —
    원격 세션에서 자동으로 uv venv 생성 + `pip install -e .[dev]`
  - `.github/workflows/ci.yml` — ruff + format + mypy + pytest (fast subset)
  - `src/common/psd_io.py::write_psd` — 진짜 per-layer PSD writer로 교체
    (psd-tools `PSDImage.new` + `create_pixel_layer`). `parse_psd`와 round-trip 가능.
  - `tests/common/test_psd_io.py`에 round-trip 테스트 추가. **32/32 pytest + 0 mypy errors**
  - dataclass slots 패턴 정리 (`field(default=None, init=False)`)
  - `scripts/bench.py` — autoresearch 속도 벤치마크 러너 (stage1/2/3 서브커맨드, TSV 기록)
  - `scripts/prepare_rife.py` — RIFE ONNX 검증/smoke-test 유틸
  - MLX SD 예제 분석 → `see-through-porting.md`에 MLX 클래스 매핑 표
  - PachiPakuGen `extract_neck_mask.py` 분석 → SAM3 **text prompt 방식** 발견,
    `sam3-rife.md` 업데이트, `neck_extractor.py`/`mouth_extractor.py`에
    리팩토링 TODO 주석 추가
- [x] **Phase 1e: End-to-end 테스트 커버리지 확장** (2026-04-10)
  - `Stage2Pipeline`에 `neck_extractor` / `mouth_extractor` 주입 경로 추가
    (dataclass field로 선택적 DI)
  - `src/stage3_rife/interpolator.py`에 `RIFEInterpolatorLike` Protocol 정의
  - `generate_eye_blink_frames` / `generate_mouth_frames` 타입 힌트를
    `RIFEInterpolatorLike`로 업그레이드 → 실제 ONNX session 없이도
    프로토콜 만족 mock으로 테스트 가능
  - 테스트 20개 추가:
    - `tests/stage2_sam3/test_pipeline.py` (3 tests) — synthetic PSD
      + mock SAM3 backend로 end-to-end. body/hair/hair_back/mouth 쌍
      파일 생성 검증, hair_back 없는 경우 edge case
    - `tests/stage3_rife/test_frame_generators.py` (7 tests) —
      MockRIFEInterpolator로 eye_blink + mouth_frames. 프레임 수,
      시작/끝 매칭, nested output 디렉토리 생성, 모음 누락 skip
    - `tests/test_cli.py` (10 tests) — typer CliRunner로 info/stage1/
      stage2/stage3/run 명령 help + exit code 검증. `layra info`는
      실제 Rich 테이블 출력까지 확인.
  - **`make ci`: 115/115 pytest, 0 mypy errors, 34 source files**
- [x] **Phase 1d: Stage 2 text-prompt 리팩토링** (2026-04-10)
  - `src/stage2_sam3/sam3_backend.py` (new) — `Sam3TextExtractor` +
    `combine_masks` + `postprocess_mask` + `extract_with_processor` 순수 함수 +
    `Sam3ProcessorLike` Protocol
  - `NeckExtractor`/`MouthExtractor`를 Sam3TextExtractor 래퍼로 리팩토링
    (composition). `extract()` 기본 경로는 text prompt.
  - Legacy point-based API는 `NeckExtractor.extract_from_point`로만 보존
  - `Stage2Config` 확장:
    - `confidence_threshold=0.3` (PachiPakuGen 기본값)
    - `text_prompts` dict (기본: neck/mouth/eye)
    - `postprocess_dilate_iterations=2`, `postprocess_blur_kernel=7`
    - `bpe_path: Path | None` (SAM3 BPE 토크나이저 오버라이드)
  - 테스트 21개 추가 (`tests/stage2_sam3/`):
    - `MockSam3Processor`로 Sam3ProcessorLike protocol 구현
    - combine_masks / postprocess_mask 수식 검증
    - extract_with_processor 호출 시퀀스 검증
    - NeckExtractor/MouthExtractor end-to-end (SAM3 없이)
  - **`make ci`: 95/95 pytest, 0 mypy errors, 34 source files**
  - 맥북 도착 시 SAM3 weights만 있으면 lazy load로 바로 동작
- [x] **Phase 1c: MLX 포팅 템플릿 프리뷰** (2026-04-10)
  - MLX SD 예제(`unet.py`, `vae.py`, `sampler.py`, `config.py`)를 WebFetch로
    상세 정독 → 정확한 class/__init__/config 시그니처 확보
  - `src/stage1_layerdiff/configs.py` — SDXL-LayerDiffuse 전체 dataclass config
    (UNetFrameConditionConfig, TransparentVAEConfig, SDXLTextEncoderConfig,
    MarigoldConfig, DiffusionConfig). 전부 `frozen=True`.
  - `src/stage1_layerdiff/mlx_ops/` — 6개 프리미티브 모듈 (embeddings, norms,
    attention, resnet, blocks) + CrossFrameTransformerBlock, Transformer3DModel
  - `src/stage1_layerdiff/unet_frame.py` — UNetFrameConditionModel 스켈레톤,
    전체 forward pseudo-code가 docstring에 상세히 기록됨
  - `src/stage1_layerdiff/vae.py` — TransparentVAE 스켈레톤
  - `src/stage1_layerdiff/schedulers.py` — **완전 구현된 DPM++ 2M SDE** (numpy 기반).
    beta/alphas_cumprod/Karras sigmas/single step 업데이트 모두 수학적으로 검증됨
  - `src/stage1_layerdiff/weights.py` — PyTorch safetensors → MLX 변환 완전 스캐폴딩
    (component별 KEY_MAPS, REWRITE_RULES, CLI)
  - `src/stage1_layerdiff/model.py`/`marigold.py` — 새 모듈들 사용, sample()/
    predict_depth() 의사코드가 docstring에 저장됨
  - 테스트 추가:
    - `tests/stage1_layerdiff/test_configs.py` (12 tests)
    - `tests/stage1_layerdiff/test_schedulers.py` (20 tests — 수식 검증)
    - `tests/stage1_layerdiff/test_weights_mapping.py` (10 tests)
  - **`make ci` 결과: 74/74 pytest, 0 mypy errors across 33 files**
  - 맥북 도착 시 `NotImplementedError` 위치만 채우면 Stage 1이 동작하는 상태

---

## 맥북 도착 전 할 것

- [x] See-Through 코드 전체 정독 (inference_psd.py, transformer3d, inference_utils) — Phase 1a/1b
- [x] PachiPakuGen scripts/ 디렉토리 정독 — Phase 1b (Python 파일 1개뿐)
- [ ] LayerDiffuse 원본 코드 정독 (별도 레포, diffusers 모듈과 중복 정독해도 OK)
- [ ] Marigold 원본 코드 정독 (prs-eth/marigold)
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
