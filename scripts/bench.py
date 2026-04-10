"""autoresearch 메트릭 벤치마크 러너.

단일 바이너리로 실행 시간 측정 + TSV 기록을 수행한다.

사용 예::

    # Stage 1 속도 측정 (baseline)
    python scripts/bench.py stage1 \\
        --experiment-id baseline \\
        --parameter-changed baseline --value - \\
        --notes "initial M5 Pro run"

    # 파라미터 변경 후 측정
    python scripts/bench.py stage1 \\
        --experiment-id exp-001 \\
        --parameter-changed num_inference_steps --value 20 \\
        --notes "try 20 steps"

출력은 `experiments/results/speed_optimization/results.tsv`에 한 줄 추가.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

# 프로젝트 루트를 sys.path에 삽입해 `python scripts/bench.py`로 직접 실행 가능
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


# ============================================================
# 결과 레코드 & TSV 기록
# ============================================================


@dataclasses.dataclass(slots=True)
class BenchRecord:
    """experiments/results/<metric>/results.tsv 한 행."""

    experiment_id: str
    timestamp: str
    parameter_changed: str
    value: str
    metric: float  # 초 (speed) 또는 IoU (quality)
    delta: str  # "-", "+3.1s", "-0.02" 등
    status: str  # baseline / keep / revert / fail
    notes: str

    def to_tsv_line(self) -> str:
        fields = [
            self.experiment_id,
            self.timestamp,
            self.parameter_changed,
            self.value,
            f"{self.metric:.3f}",
            self.delta,
            self.status,
            self.notes.replace("\t", " ").replace("\n", " "),
        ]
        return "\t".join(fields) + "\n"


def append_record(tsv_path: Path, record: BenchRecord) -> None:
    """TSV에 한 줄 추가. 헤더 없으면 생성."""
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "experiment_id\ttimestamp\tparameter_changed\tvalue"
        "\tmetric\tdelta\tstatus\tnotes\n"
    )
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        tsv_path.write_text(header, encoding="utf-8")
    with tsv_path.open("a", encoding="utf-8") as f:
        f.write(record.to_tsv_line())


def read_last_baseline(tsv_path: Path) -> float | None:
    """가장 최근 baseline/keep 레코드의 metric 값을 반환."""
    if not tsv_path.exists():
        return None
    lines = tsv_path.read_text(encoding="utf-8").strip().splitlines()
    for line in reversed(lines[1:]):  # skip header
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        status = fields[6]
        if status in {"baseline", "keep"}:
            try:
                return float(fields[4])
            except ValueError:
                continue
    return None


# ============================================================
# 시간 측정 헬퍼
# ============================================================


def _measure(
    runner: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, list[float]]:
    """runner를 warmup + repeats 번 실행하고 중앙값을 반환.

    Returns:
        (median_seconds, all_seconds)
    """
    for _ in range(warmup):
        runner()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        runner()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples


# ============================================================
# Stage 별 벤치마크
# ============================================================


def bench_stage1(images: Sequence[Path], repeats: int, warmup: int) -> float:
    """Stage 1 (LayerDiffuse) 이미지별 평균 처리 시간 (초).

    MLX 포팅이 미완료인 동안은 `NotImplementedError`를 catch해 명시적인
    에러 메시지로 fail한다.
    """
    from src.stage1_layerdiff import run_stage1

    if not images:
        raise ValueError("No input images provided")

    def make_runner(img: Path, psd: Path) -> Callable[[], None]:
        def runner() -> None:
            run_stage1(image_path=img, psd_output=psd)

        return runner

    per_image_times: list[float] = []
    for img_path in images:
        out_psd = img_path.with_suffix(".out.psd")
        try:
            median, _samples = _measure(
                make_runner(img_path, out_psd),
                repeats=repeats,
                warmup=warmup,
            )
        except NotImplementedError as e:
            logger.error(f"Stage 1 not ready: {e}")
            raise
        logger.info(f"Stage 1 on {img_path.name}: {median:.2f}s (median)")
        per_image_times.append(median)

    return statistics.mean(per_image_times)


def bench_stage2(
    original_image: Path, psd: Path, output_dir: Path, repeats: int, warmup: int
) -> float:
    """Stage 2 (SAM3 보정) 단일 실행 시간 (초)."""
    from src.stage2_sam3 import run_stage2

    def runner() -> None:
        run_stage2(
            original_image=original_image,
            psd_path=psd,
            output_dir=output_dir,
        )

    median, _ = _measure(runner, repeats=repeats, warmup=warmup)
    return median


def bench_stage3(
    eye_open: Path,
    eye_closed: Path,
    mouth_closed: Path,
    mouth_a: Path,
    output_dir: Path,
    repeats: int,
    warmup: int,
) -> float:
    """Stage 3 (RIFE) 단일 실행 시간 (초) — 눈 + 1모음만."""
    from src.stage3_rife import run_stage3
    from src.stage3_rife.pipeline import Stage3Inputs

    inputs = Stage3Inputs(
        eye_open=eye_open,
        eye_closed=eye_closed,
        mouth_closed=mouth_closed,
        mouth_vowels={"a": mouth_a},
    )

    def runner() -> None:
        run_stage3(inputs=inputs, output_dir=output_dir)

    median, _ = _measure(runner, repeats=repeats, warmup=warmup)
    return median


# ============================================================
# CLI
# ============================================================


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--parameter-changed", required=True)
    p.add_argument("--value", default="-")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--notes", default="")
    p.add_argument(
        "--status",
        choices=["baseline", "keep", "revert", "fail"],
        default=None,
        help=(
            "명시하지 않으면 직전 baseline/keep과 비교해 개선되면 keep, "
            "아니면 revert로 기록."
        ),
    )


def _resolve_status(
    provided: str | None, current: float, baseline: float | None
) -> tuple[str, str]:
    """status + delta 문자열 결정."""
    if baseline is None:
        return (provided or "baseline", "-")
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    delta_str = f"{sign}{delta:.2f}"
    if provided is not None:
        return (provided, delta_str)
    # auto: 낮을수록 좋다고 가정 (시간 메트릭)
    return ("keep" if delta < 0 else "revert", delta_str)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layra autoresearch bench runner")
    sub = parser.add_subparsers(dest="stage", required=True)

    # stage1
    p1 = sub.add_parser("stage1", help="Stage 1 레이어 분해 속도 측정")
    p1.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=None,
        help="테스트 이미지들. 기본값: experiments/test_images/*.png",
    )
    _common_args(p1)

    # stage2
    p2 = sub.add_parser("stage2", help="Stage 2 SAM3 보정 속도 측정")
    p2.add_argument("--image", type=Path, required=True)
    p2.add_argument("--psd", type=Path, required=True)
    p2.add_argument("--output-dir", type=Path, required=True)
    _common_args(p2)

    # stage3
    p3 = sub.add_parser("stage3", help="Stage 3 RIFE 프레임 보간 속도 측정")
    p3.add_argument("--eye-open", type=Path, required=True)
    p3.add_argument("--eye-closed", type=Path, required=True)
    p3.add_argument("--mouth-closed", type=Path, required=True)
    p3.add_argument("--mouth-a", type=Path, required=True)
    p3.add_argument("--output-dir", type=Path, required=True)
    _common_args(p3)

    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging(level="INFO")

    tsv = PROJECT_ROOT / "experiments/results/speed_optimization/results.tsv"
    baseline = read_last_baseline(tsv)

    try:
        if args.stage == "stage1":
            images = args.images or sorted(
                (PROJECT_ROOT / "experiments/test_images").glob("*.png")
            )
            metric = bench_stage1(images, args.repeats, args.warmup)
        elif args.stage == "stage2":
            metric = bench_stage2(
                args.image, args.psd, args.output_dir, args.repeats, args.warmup
            )
        elif args.stage == "stage3":
            metric = bench_stage3(
                args.eye_open,
                args.eye_closed,
                args.mouth_closed,
                args.mouth_a,
                args.output_dir,
                args.repeats,
                args.warmup,
            )
        else:
            parser.error(f"Unknown stage: {args.stage}")
            return 2
    except Exception as e:
        record = BenchRecord(
            experiment_id=args.experiment_id,
            timestamp=dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
            parameter_changed=args.parameter_changed,
            value=str(args.value),
            metric=-1.0,
            delta="-",
            status="fail",
            notes=f"{type(e).__name__}: {e}",
        )
        append_record(tsv, record)
        logger.error(f"Bench failed: {e}")
        return 1

    status, delta_str = _resolve_status(args.status, metric, baseline)
    record = BenchRecord(
        experiment_id=args.experiment_id,
        timestamp=dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        parameter_changed=args.parameter_changed,
        value=str(args.value),
        metric=metric,
        delta=delta_str,
        status=status,
        notes=args.notes,
    )
    append_record(tsv, record)

    baseline_str = f"{baseline:.3f}" if baseline is not None else "—"
    logger.info(
        f"[{args.stage}] metric={metric:.3f}s "
        f"(baseline={baseline_str}) "
        f"delta={delta_str} status={status}"
    )
    logger.info(f"Appended to {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
