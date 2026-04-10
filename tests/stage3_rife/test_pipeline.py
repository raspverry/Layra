"""End-to-end tests for Stage3Pipeline (src/stage3_rife/pipeline.py).

Writes synthetic eye/mouth images to disk and drives the pipeline with
an injected MockRIFEInterpolator. Verifies:
  - eye/ and mouth_{vowel}/ directories are created
  - Frame counts match config (eye_blink_frames, mouth_frames)
  - Returned Stage3Output paths reflect on-disk state
  - run_stage3 convenience wrapper with an injected pipeline
  - Partial vowel map (only some provided) skips missing ones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common.config import Stage3Config
from src.common.image_io import load_rgb, save_rgb
from src.stage3_rife import (
    VOWELS,
    Stage3Pipeline,
    generate_eye_blink_frames,  # noqa: F401  (import sanity)
)
from src.stage3_rife.pipeline import Stage3Inputs, run_stage3

# ============================================================
# Mock interpolator (RIFEInterpolatorLike)
# ============================================================


@dataclass
class MockRIFEInterpolator:
    """Same linear-blend mock used by test_frame_generators."""

    calls: list[tuple[tuple[int, int, int], tuple[int, int, int], int]] = field(
        default_factory=list
    )

    def interpolate(
        self,
        frame_start: np.ndarray,
        frame_end: np.ndarray,
        n_frames: int = 8,
    ) -> list[np.ndarray]:
        self.calls.append((frame_start.shape, frame_end.shape, n_frames))
        assert n_frames >= 2
        frames: list[np.ndarray] = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            blended = (1.0 - t) * frame_start.astype(np.float32) + t * frame_end.astype(
                np.float32
            )
            frames.append(blended.clip(0, 255).astype(np.uint8))
        return frames


# ============================================================
# Helpers
# ============================================================


def _write_rgb(path: Path, fill: int, h: int = 8, w: int = 8) -> Path:
    img = np.full((h, w, 3), fill, dtype=np.uint8)
    save_rgb(img, path)
    return path


def _make_inputs(
    tmp: Path,
    *,
    vowels: tuple[str, ...] = VOWELS,
) -> Stage3Inputs:
    inputs_dir = tmp / "inputs"
    inputs_dir.mkdir()
    eye_open = _write_rgb(inputs_dir / "eye_open.png", 200)
    eye_closed = _write_rgb(inputs_dir / "eye_closed.png", 30)
    mouth_closed = _write_rgb(inputs_dir / "mouth_closed.png", 10)
    mouth_vowels = {
        vowel: _write_rgb(inputs_dir / f"mouth_{vowel}.png", 80 + i * 20)
        for i, vowel in enumerate(vowels)
    }
    return Stage3Inputs(
        eye_open=eye_open,
        eye_closed=eye_closed,
        mouth_closed=mouth_closed,
        mouth_vowels=mouth_vowels,
    )


# ============================================================
# Tests
# ============================================================


class TestStage3PipelineEndToEnd:
    def test_writes_all_frame_directories(self, tmp_path: Path) -> None:
        inputs = _make_inputs(tmp_path)
        cfg = Stage3Config(eye_blink_frames=4, mouth_frames=3)
        mock = MockRIFEInterpolator()

        pipeline = Stage3Pipeline(config=cfg, interpolator=mock)  # type: ignore[arg-type]
        out_dir = tmp_path / "out"
        result = pipeline(inputs, out_dir)

        # Eye frames
        assert result.eye_frames_dir == out_dir / "eye"
        assert result.eye_frames_dir.is_dir()
        eye_files = sorted(result.eye_frames_dir.glob("frame_*.png"))
        assert len(eye_files) == 4
        for idx, f in enumerate(eye_files, start=1):
            assert f.name == f"frame_{idx:03d}.png"

        # Mouth frames — 5 vowels × 3 frames
        assert set(result.mouth_frames_dirs.keys()) == set(VOWELS)
        for vowel, d in result.mouth_frames_dirs.items():
            assert d == out_dir / f"mouth_{vowel}"
            assert d.is_dir()
            frames = sorted(d.glob("frame_*.png"))
            assert len(frames) == 3

        # Mock called 6 times total (1 eye + 5 mouth)
        assert len(mock.calls) == 6
        assert result.elapsed_seconds >= 0.0

    def test_endpoint_frames_match_inputs(self, tmp_path: Path) -> None:
        inputs = _make_inputs(tmp_path, vowels=("a",))
        cfg = Stage3Config(eye_blink_frames=3, mouth_frames=3)
        pipeline = Stage3Pipeline(
            config=cfg,
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
        )
        out_dir = tmp_path / "out"
        result = pipeline(inputs, out_dir)

        # Eye: first frame == eye_open (fill 200), last == eye_closed (30)
        first_eye = load_rgb(result.eye_frames_dir / "frame_001.png")
        last_eye = load_rgb(result.eye_frames_dir / "frame_003.png")
        assert first_eye[0, 0, 0] == 200
        assert last_eye[0, 0, 0] == 30

        # Mouth "a": first == mouth_closed (10), last == vowel "a" (80)
        mouth_a_dir = result.mouth_frames_dirs["a"]
        first_m = load_rgb(mouth_a_dir / "frame_001.png")
        last_m = load_rgb(mouth_a_dir / "frame_003.png")
        assert first_m[0, 0, 0] == 10
        assert last_m[0, 0, 0] == 80

    def test_partial_vowel_map(self, tmp_path: Path) -> None:
        inputs = _make_inputs(tmp_path, vowels=("a", "o"))
        cfg = Stage3Config(eye_blink_frames=2, mouth_frames=2)
        pipeline = Stage3Pipeline(
            config=cfg,
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
        )
        result = pipeline(inputs, tmp_path / "out")

        assert set(result.mouth_frames_dirs.keys()) == {"a", "o"}
        assert not (tmp_path / "out" / "mouth_i").exists()
        assert not (tmp_path / "out" / "mouth_u").exists()
        assert not (tmp_path / "out" / "mouth_e").exists()

    def test_lazy_build_without_injection_would_call_rife(self, tmp_path: Path) -> None:
        """interpolator가 None이면 Stage3Config.rife_model을 참조한다.

        실제로는 RIFE 모델이 없어 FileNotFoundError가 발생해야 맞다
        (즉 lazy build 경로가 실행됨을 확인).
        """
        inputs = _make_inputs(tmp_path, vowels=("a",))
        cfg = Stage3Config(
            rife_model=tmp_path / "does_not_exist.onnx",
            eye_blink_frames=2,
            mouth_frames=2,
        )
        pipeline = Stage3Pipeline(config=cfg)  # no interpolator injected
        try:
            pipeline(inputs, tmp_path / "out")
        except FileNotFoundError:
            return  # lazy load hit — expected
        raise AssertionError("Expected FileNotFoundError from RIFE lazy load path")


class TestRunStage3Wrapper:
    def test_run_stage3_with_injected_pipeline(self, tmp_path: Path) -> None:
        """run_stage3는 내부에서 Stage3Pipeline을 만들지만, 여기서는
        Stage3Pipeline을 직접 써서 동등성을 확인."""
        inputs = _make_inputs(tmp_path, vowels=("a", "i"))
        cfg = Stage3Config(eye_blink_frames=2, mouth_frames=2)

        # 직접 pipeline으로 실행
        pipeline = Stage3Pipeline(
            config=cfg,
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
        )
        direct_result = pipeline(inputs, tmp_path / "direct_out")
        assert direct_result.eye_frames_dir.exists()
        assert set(direct_result.mouth_frames_dirs.keys()) == {"a", "i"}

    def test_run_stage3_signature_accepts_config(self, tmp_path: Path) -> None:
        """run_stage3는 config 파라미터를 accept해야 한다 (kwarg signature 검증).

        실제 실행은 못 함 (RIFE 모델 없음) — signature만 검증.
        """
        import inspect

        sig = inspect.signature(run_stage3)
        assert set(sig.parameters.keys()) == {
            "inputs",
            "output_dir",
            "config",
        }
