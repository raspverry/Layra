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

## ADR-007: 프로젝트 라이선스 → Apache 2.0

**날짜**: 2026-04-10
**결정**: Layra 본인 코드는 Apache 2.0으로 배포

**이유**:
- 주요 참조 레포 중 상당수가 Apache 2.0 (See-Through, LayerDiffuse, Marigold, SAM)
- 라이선스 호환성 단순화 (특히 See-Through 기반 derivative work)
- PachiPakuGen은 MIT라 Apache 2.0에 호환됨
- 특허 조항으로 기여자 보호
- 서비스화 시에도 본인 코드는 동일 라이선스 유지 가능

**대안**: MIT
**기각 이유**: 파생 저작물 범위가 넓어 특허 조항이 있는 Apache 2.0이 안전

**주의**: 이것은 **Layra 본인 코드**의 라이선스일 뿐, 모델 weights의 라이선스는
BLOCKER-001에 따라 별도 확인 필요.

---

## ADR-008: 초기 코드 구조 → src/common + stage 모듈 분리

**날짜**: 2026-04-10
**결정**:
- `src/common/` — 설정(Pydantic), 로깅(loguru), 타입, 이미지/PSD I/O
- `src/stage1_layerdiff/`, `src/stage2_sam3/`, `src/stage3_rife/` — Stage별 파이프라인
- `src/cli.py` — typer 기반 단일 엔트리포인트
- Stage 간 의존성은 파일(PSD, PNG)로만 (in-memory 결합 없음)

**이유**:
- Stage마다 필요한 런타임(MLX/PyTorch MPS/ONNX)이 다름 → import 격리 필요
- 파일 기반 전달이면 각 Stage를 독립적으로 테스트/디버깅 가능
- RunPod Serverless에서 Stage별 분리 배포 여지 확보
- CLI 명령도 stage1 / stage2 / stage3 / run 분리 → 부분 실행 가능

**결과**:
- `segment_anything_3`, `mlx`, `onnxruntime` import는 전부 함수 내부에서 lazy
- Stage 2/3 일부는 MLX 없어도 psd-tools/onnxruntime만으로 개발 가능

---

## ADR-009: Claude Code 세션 하네스 → sync SessionStart hook + uv

**날짜**: 2026-04-10 (Phase 1b)
**결정**:
- `.claude/hooks/session-start.sh` + `.claude/settings.json`으로
  SessionStart hook 등록
- `CLAUDE_CODE_REMOTE=true`일 때만 실행 (로컬 Mac 세션은 건드리지 않음)
- `uv venv`로 Python 3.12 가상환경 생성 → `uv pip install -e ".[dev]"`
- `$CLAUDE_ENV_FILE`에 `VIRTUAL_ENV`/`PATH`/`PYTHONPATH` export
- **sync 모드** (async 아님)

**이유**:
- 원격 컨테이너에서 매 세션마다 `make ci`가 바로 돌아야 개발 속도 유지
- `uv`가 pip보다 5~10배 빨라 초기 설치 시간 단축
- sync 모드: dependencies가 반드시 설치된 상태에서 세션 시작
  → Claude가 pytest/ruff를 실행할 때 race condition 없음
- `CLAUDE_CODE_REMOTE` 가드: Mac 개발 환경에서는 충돌 없음

**대안**:
- async 모드 (세션 부팅 빠름, race condition 있음)
- Dockerfile + devcontainer (복잡도 증가)
- 수동 setup.sh (매 세션마다 반복)

**기각 이유**:
- async: 첫 세션에서 `make ci` 실행 시 setuptools/torch 설치 중이면 실패
- Docker: `.claude/hooks/` 표준 패턴이 더 깔끔
- 수동: CLAUDE.md의 "세션 시작 시 필수" 규칙과 상충

**결과**:
- 원격 컨테이너에서 95+ pytest + mypy + ruff가 "세션 열자마자" 실행 가능
- 초회 설치 후 venv 캐시되어 재부팅 시 수 초 내 완료
- CLAUDE.md의 "세션 시작 시 읽을 것" 규칙이 자연스럽게 동작
- Mac 도착 후에도 `CLAUDE_CODE_REMOTE` 가드 덕분에 간섭 없음

---

## ADR-010: SAM3 호출 방식 → text prompt (point prompt 아님)

**날짜**: 2026-04-10 (Phase 1b 분석 → Phase 1d 구현)
**결정**:
- `src/stage2_sam3/sam3_backend.py`의 `Sam3TextExtractor`가
  `build_sam3_image_model` + `Sam3Processor.set_text_prompt` 패턴 사용
- 기본 프롬프트: `{"neck": "neck", "mouth": "mouth", "eye": "eye"}`
- `confidence_threshold=0.3` (PachiPakuGen과 동일)
- 마스크 후처리: `cv2.dilate(2회)` + `cv2.GaussianBlur(7×7)`
- 레거시 point-based 경로는 `NeckExtractor.extract_from_point`로만 보존

**이유**:
- PachiPakuGen의 `scripts/extract_neck_mask.py`를 Phase 1b에 WebFetch로
  분석한 결과, 저자가 실제로 point 대신 text prompt를 쓴다는 점을 발견.
- Text prompt는 좌표 추정 없이 원본 이미지만으로 동작 → 얼굴 bbox
  자동 검출 로직 불필요
- SAM3의 강점(언어 prompt)을 사용하는 쪽이 레이어 품질이 더 좋다는
  PachiPakuGen 실증 결과를 따름
- Config의 `text_prompts` dict로 프롬프트 커스터마이징 가능 (e.g.
  `"neck,throat"` 같은 복합 프롬프트)

**대안**:
- Point prompt (Phase 0 초기 스캐폴딩 기본값)
- Box prompt
- Automatic masks (SAM의 `generate()` mode)

**기각 이유**:
- Point prompt: 얼굴 검출 선행 필요, 추정 실패 시 오동작
- Box prompt: 수동 조정 필요, 자동화 어려움
- Automatic: 레이어 이름 의미 매핑 불가능

**결과**:
- `NeckExtractor.extract(image)` 한 줄로 깔끔한 neck 마스크
- Linux CI에서 `Sam3ProcessorLike` mock을 통해 전체 경로 테스트 가능
- Mac 도착 후 `segment_anything_3` 패키지만 설치되면 즉시 동작

---

## ADR-011: Backend 구조적 타이핑 → Protocol 기반 DI

**날짜**: 2026-04-10 (Phase 1d + Phase 1e)
**결정**:
- 무거운 외부 의존성을 가진 컴포넌트에는 `typing.Protocol`을 정의해
  structural typing으로 DI:
  - `Sam3ProcessorLike` (Phase 1d) — SAM3 processor
  - `RIFEInterpolatorLike` (Phase 1e) — RIFE ONNX interpolator
- `Sam3TextExtractor`, `Stage2Pipeline`, `Stage3Pipeline`에
  `inject_*` / `interpolator=...` 주입 경로 추가
- 주입하지 않으면 config로부터 lazy build (backward compatible)

**이유**:
- SAM3 weights(3.2GB)와 RIFE ONNX(~25MB)를 **Linux CI에 두지 않아도**
  오케스트레이션 경로 전부 테스트 가능
- Protocol은 상속 없이도 구조만 맞으면 되므로 테스트 mock이 가볍다
  (class 10줄로 Sam3Processor 행동 복제)
- 실제 `Sam3Processor` / `RIFEInterpolator`도 자연스럽게 Protocol 만족
  → production 코드 수정 불필요
- Phase 1e에서 추가한 Stage2/Stage3 integration test가 이 패턴 덕에
  20개나 mlx/SAM3/RIFE 없이 돌아간다

**대안**:
- ABC (추상 클래스) 기반 상속
- 글로벌 monkey patch로 class 교체
- `unittest.mock.Mock(spec=...)`

**기각 이유**:
- ABC: 상속 강제 → 외부 라이브러리(`Sam3Processor`)가 상속 안 해 깨짐
- Monkey patch: 테스트 상호 간섭 위험
- `Mock(spec=...)`: 타입 체커가 모름, 리팩토링 리스크

**결과**:
- **95 → 115 tests** (+20 mock-based E2E), 모두 0.9초 내 완료
- `mypy` strict 모드에서도 Protocol 방식은 문제 없음
- Mac 도착 후 실제 backend 주입하면 동일 경로로 production 동작

---

## ADR-012: MLX 포팅 템플릿의 타입 전략 → `MLXArray = Any`

**날짜**: 2026-04-10 (Phase 1c)
**결정**:
- `src/stage1_layerdiff/mlx_ops/*.py` 및 `unet_frame.py`, `vae.py`,
  `model.py`, `marigold.py`의 MLX 관련 타입 alias는 **모듈 탑레벨**에
  `MLXArray = Any`로 통일
- `mlx.core`, `mlx.nn` import는 모두 함수 내부 (lazy) 또는
  `_require_mlx()` 가드 뒤에 둠
- `from __future__ import annotations` 필수

**이유**:
- MLX는 `platform_system == 'Darwin' and platform_machine == 'arm64'`
  에서만 설치됨 → Linux CI에서는 설치 자체 불가능
- `TYPE_CHECKING`으로 alias를 조건부 선언하면 mypy가
  "Variable is not valid as a type" 오류를 냄 (두 할당 중 하나를
  type alias로 인식 못함)
- `Any`는 "아직 구현 안 됨"이 아니라 "Linux 환경에서는 검증 불가능한
  MLX 배열"이라는 명시적 표현
- 실제 구현 시 Mac에서는 `mlx.core.array` 객체가 들어오지만 Any는
  duck-typing을 허용하므로 작동에 영향 없음

**대안**:
- `TYPE_CHECKING` 조건부 alias (`MLXArray = mx.array if TYPE_CHECKING else Any`)
- `TypeAlias` + string literal
- mlx 설치된 CI (예: macOS runner on GitHub Actions)

**기각 이유**:
- 조건부 alias: mypy 에러 (확인된 문제)
- TypeAlias string: Python 3.12의 PEP 695 `type` statement가 더 나은
  대안이지만 mlx가 런타임에 없으면 여전히 평가 안 됨
- macOS runner: CI 시간/비용 증가, 매 PR에서 GPU 없이 MLX 실행 불가

**결과**:
- Linux CI (`ubuntu-latest`)에서 mypy가 mlx 관련 파일 33개 전부 통과
- Mac 도착 후 `MLXArray = mx.array`로 한 번에 교체 가능 (sed 한 줄)
- 현재 115 tests 중 mlx 필요 테스트는 0개 — 전부 Linux에서 돌아감

---

## 추가 결정 사항

(작업하면서 새로운 결정이 생기면 여기에 추가)
