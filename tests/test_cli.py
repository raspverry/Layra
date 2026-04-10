"""Smoke tests for the `layra` CLI (src/cli.py).

Uses typer's CliRunner to drive the app without spawning subprocesses.
We don't run the ML pipelines themselves (they raise NotImplementedError
until MLX port lands); we just verify that command wiring, argument
parsing, and the `info` table print path all work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli import app


@pytest.fixture
def runner() -> CliRunner:
    # mix_stderr removed in typer 0.12+; CliRunner defaults are fine.
    return CliRunner()


class TestTopLevel:
    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # typer help 출력에 command 목록이 포함되어야 함
        assert "info" in result.output
        assert "stage1" in result.output
        assert "stage2" in result.output
        assert "stage3" in result.output
        assert "run" in result.output

    def test_no_args_shows_help(self, runner: CliRunner) -> None:
        """`no_args_is_help=True` 설정 검증."""
        result = runner.invoke(app, [])
        # Typer shows help and exits with 0 (or 2) — accept both
        assert result.exit_code in (0, 2)
        assert "Layra" in result.output or "info" in result.output


class TestInfoCommand:
    def test_info_prints_config_table(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0, result.output
        assert "Layra Configuration" in result.output
        # 주요 설정 키들이 출력에 있어야 함
        assert "project_root" in result.output
        assert "stage1.dtype" in result.output
        assert "stage2.sam3_weights" in result.output
        assert "stage3.rife_model" in result.output

    def test_info_with_verbose_flag(self, runner: CliRunner) -> None:
        """글로벌 --verbose 옵션이 exit code를 건드리지 않아야 함."""
        result = runner.invoke(app, ["--verbose", "info"])
        assert result.exit_code == 0


class TestStage1Command:
    def test_stage1_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["stage1", "--help"])
        assert result.exit_code == 0
        assert "image" in result.output.lower()
        assert "--output" in result.output or "-o" in result.output

    def test_stage1_missing_image_errors(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["stage1"])
        # Missing required argument
        assert result.exit_code != 0


class TestStage2Command:
    def test_stage2_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["stage2", "--help"])
        assert result.exit_code == 0
        assert "image" in result.output.lower() or "IMAGE" in result.output
        assert "psd" in result.output.lower() or "PSD" in result.output


class TestStage3Command:
    def test_stage3_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["stage3", "--help"])
        assert result.exit_code == 0
        assert "--eye-open" in result.output
        assert "--eye-closed" in result.output
        assert "--mouth-closed" in result.output
        # 모음 옵션들
        for vowel in ("a", "i", "u", "e", "o"):
            assert f"--mouth-{vowel}" in result.output


class TestRunCommand:
    def test_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_exits_with_stub_code(self, runner: CliRunner, tmp_path: Path) -> None:
        """full pipeline은 아직 Stage 1 MLX 미완성이라 exit=2."""
        # 존재하지 않는 이미지여도 typer는 Argument로 Path를 받아들임
        dummy = tmp_path / "image.png"
        dummy.write_bytes(b"not a real png")
        result = runner.invoke(app, ["run", str(dummy)])
        # Stub이 typer.Exit(code=2)를 던짐
        assert result.exit_code == 2
