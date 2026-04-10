"""RIFE ONNX 모델 준비/검증 스크립트.

Layra는 RIFE ONNX를 번들하지 않는다. 사용자가 직접 다운로드해서
`models/rife.onnx`에 두어야 한다. 이 스크립트는:

1. 모델 파일 존재/크기 검증
2. ONNX Runtime execution providers 확인
3. 더미 입력으로 smoke test (실제 모델 입력 이름 추출)

사용 예::

    python scripts/prepare_rife.py --model models/rife.onnx
    python scripts/prepare_rife.py --model models/rife.onnx --smoke-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

# RIFE v4.x ONNX 파일은 보통 10~30MB 사이 (PachiPakuGen은 23MB).
# 정확한 크기는 변형(RIFE v4.9, v4.14 등)마다 다르다.
_MIN_BYTES = 5 * 1024 * 1024
_MAX_BYTES = 200 * 1024 * 1024


def check_file(model_path: Path) -> None:
    """파일 존재 + 크기 sanity."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"RIFE ONNX not found: {model_path}\n"
            "Download from a trusted source (e.g. rife-onnx releases) and "
            "place it at the path above."
        )
    size = model_path.stat().st_size
    if size < _MIN_BYTES or size > _MAX_BYTES:
        logger.warning(
            f"Unusual RIFE model size: {size / 1e6:.1f} MB "
            f"(expected {_MIN_BYTES / 1e6:.0f}-{_MAX_BYTES / 1e6:.0f} MB)"
        )
    else:
        logger.info(f"RIFE model size: {size / 1e6:.1f} MB — OK")


def list_providers() -> list[str]:
    """사용 가능한 ONNX Runtime providers."""
    import onnxruntime as ort

    providers: list[str] = list(ort.get_available_providers())
    logger.info(f"ONNX Runtime version: {ort.__version__}")
    logger.info(f"Available providers: {providers}")

    if "CoreMLExecutionProvider" in providers:
        logger.info("✅ CoreMLExecutionProvider available (Mac)")
    elif "CUDAExecutionProvider" in providers:
        logger.info("✅ CUDAExecutionProvider available")
    else:
        logger.warning("⚠️  Only CPU available. Stage 3 will be slower than targets.")
    return providers


def inspect_graph(model_path: Path) -> None:
    """모델의 입력/출력 이름과 shape 출력."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],  # 스키마 확인용
    )
    logger.info("== RIFE ONNX graph ==")
    for inp in session.get_inputs():
        logger.info(f"  input  {inp.name}: {inp.shape} {inp.type}")
    for out in session.get_outputs():
        logger.info(f"  output {out.name}: {out.shape} {out.type}")


def smoke_test(model_path: Path) -> None:
    """64x64 더미 프레임 2장으로 1회 추론 실행."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}

    # 일반적인 RIFE ONNX 입력: img0, img1, (optional) timestep
    H, W = 64, 64
    img0 = np.zeros((1, 3, H, W), dtype=np.float32)
    img1 = np.ones((1, 3, H, W), dtype=np.float32)

    feed: dict[str, np.ndarray] = {}
    if "img0" in input_names:
        feed["img0"] = img0
    if "img1" in input_names:
        feed["img1"] = img1
    if "timestep" in input_names:
        feed["timestep"] = np.array([0.5], dtype=np.float32)

    if not feed:
        # 입력 이름이 다른 변형: 위치 기반 매핑 시도
        names = [i.name for i in session.get_inputs()]
        if len(names) >= 2:
            feed[names[0]] = img0
            feed[names[1]] = img1
        if len(names) >= 3:
            feed[names[2]] = np.array([0.5], dtype=np.float32)

    logger.info(f"Smoke test feed keys: {list(feed.keys())}")
    outputs = session.run(None, feed)
    logger.info(f"Output shapes: {[o.shape for o in outputs]}")
    logger.info("✅ RIFE smoke test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare & verify RIFE ONNX")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/rife.onnx",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny dummy inference",
    )
    args = parser.parse_args(argv)
    setup_logging(level="INFO")

    try:
        check_file(args.model)
        list_providers()
        inspect_graph(args.model)
        if args.smoke_test:
            smoke_test(args.model)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}")
        return 2

    logger.info("✅ RIFE verification complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
