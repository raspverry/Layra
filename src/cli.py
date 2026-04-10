"""Layra CLI — typer 기반.

사용 예::

    python -m src.cli info
    python -m src.cli stage1 assets/test.png --output out/test.psd
    python -m src.cli stage2 assets/test.png out/test.psd --output-dir out/
    python -m src.cli stage3 --eye-open ... --eye-closed ... --output-dir out/
    python -m src.cli run assets/test.png --output-dir out/
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.common.config import get_config
from src.common.logging import get_logger, setup_logging

app = typer.Typer(
    name="layra",
    help="애니 일러스트 → Live2D 아바타 소재 파이프라인",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="DEBUG 레벨 로깅 활성화"
    ),
) -> None:
    """전역 옵션."""
    setup_logging(level="DEBUG" if verbose else "INFO")


@app.command()
def info() -> None:
    """현재 설정과 모델 경로를 출력."""
    from rich.console import Console
    from rich.table import Table

    cfg = get_config()
    console = Console()

    table = Table(title="Layra Configuration", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("project_root", str(cfg.project_root))
    table.add_row("model_dir", str(cfg.model_dir))
    table.add_row("output_dir", str(cfg.output_dir))
    table.add_row("log_level", cfg.log_level)
    table.add_row("stage1.dtype", cfg.stage1.dtype)
    table.add_row("stage1.resolution", str(cfg.stage1.resolution))
    table.add_row("stage1.layerdiff_weights", str(cfg.stage1.layerdiff_weights))
    table.add_row("stage1.marigold_weights", str(cfg.stage1.marigold_weights))
    table.add_row("stage2.device", cfg.stage2.device)
    table.add_row("stage2.sam3_weights", str(cfg.stage2.sam3_weights))
    table.add_row("stage3.rife_model", str(cfg.stage3.rife_model))
    table.add_row("stage3.providers", ", ".join(cfg.stage3.providers))
    console.print(table)


@app.command()
def stage1(
    image: Path = typer.Argument(..., help="입력 일러스트 (PNG/JPG)"),
    output: Path = typer.Option(
        Path("out/stage1.psd"), "--output", "-o", help="PSD 출력 경로"
    ),
) -> None:
    """Stage 1만 실행: 레이어 분해 → PSD."""
    from src.stage1_layerdiff import run_stage1

    result = run_stage1(image_path=image, psd_output=output)
    typer.echo(
        f"Stage 1 done: {result.psd_path} "
        f"({len(result.layer_set)} layers, {result.elapsed_seconds:.1f}s)"
    )


@app.command()
def stage2(
    image: Path = typer.Argument(..., help="Stage 1 입력 원본 이미지"),
    psd: Path = typer.Argument(..., help="Stage 1 출력 PSD"),
    output_dir: Path = typer.Option(
        Path("out/stage2"), "--output-dir", "-d", help="출력 디렉토리"
    ),
    open_mouth: Path | None = typer.Option(
        None, "--open-mouth", help="(선택) 열린 입 참조 이미지"
    ),
) -> None:
    """Stage 2만 실행: SAM3 body/hair/mouth 보정."""
    from src.stage2_sam3 import run_stage2

    result = run_stage2(
        original_image=image,
        psd_path=psd,
        output_dir=output_dir,
        open_mouth_image=open_mouth,
    )
    typer.echo(
        f"Stage 2 done: {result.body_png.parent} ({result.elapsed_seconds:.1f}s)"
    )


@app.command()
def stage3(
    eye_open: Path = typer.Option(..., help="눈 열림 이미지"),
    eye_closed: Path = typer.Option(..., help="눈 닫힘 이미지"),
    mouth_closed: Path = typer.Option(..., help="입 닫힘 이미지"),
    mouth_a: Path | None = typer.Option(None, help="모음 あ"),
    mouth_i: Path | None = typer.Option(None, help="모음 い"),
    mouth_u: Path | None = typer.Option(None, help="모음 う"),
    mouth_e: Path | None = typer.Option(None, help="모음 え"),
    mouth_o: Path | None = typer.Option(None, help="모음 お"),
    output_dir: Path = typer.Option(
        Path("out/stage3"), "--output-dir", "-d", help="출력 디렉토리"
    ),
) -> None:
    """Stage 3만 실행: RIFE 프레임 보간."""
    from src.stage3_rife import run_stage3
    from src.stage3_rife.pipeline import Stage3Inputs

    vowel_paths: dict[str, Path] = {}
    for name, path in [
        ("a", mouth_a),
        ("i", mouth_i),
        ("u", mouth_u),
        ("e", mouth_e),
        ("o", mouth_o),
    ]:
        if path is not None:
            vowel_paths[name] = path

    inputs = Stage3Inputs(
        eye_open=eye_open,
        eye_closed=eye_closed,
        mouth_closed=mouth_closed,
        mouth_vowels=vowel_paths,
    )
    result = run_stage3(inputs=inputs, output_dir=output_dir)
    typer.echo(
        f"Stage 3 done: {result.eye_frames_dir.parent} ({result.elapsed_seconds:.1f}s)"
    )


@app.command()
def run(
    image: Path = typer.Argument(..., help="입력 일러스트"),
    output_dir: Path = typer.Option(
        Path("out/run"), "--output-dir", "-d", help="출력 루트"
    ),
) -> None:
    """전체 파이프라인 실행 (Stage 1 → 2 → 3).

    Note:
        현재 Stage 1 MLX 포팅이 미완료 상태라 이 커맨드는
        Stage1Pipeline 호출 시점에 NotImplementedError가 발생한다.
        맥북 도착 후 Stage 1 구현이 끝나면 동작한다.
    """
    logger = get_logger(__name__)
    logger.warning(
        "Full pipeline is a stub — Stage 1 MLX port pending. "
        "See wiki/see-through-porting.md"
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
