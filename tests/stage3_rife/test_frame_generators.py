"""Stage 3 frame generators with a mock RIFEInterpolator.

No RIFE ONNX model is required — MockRIFEInterpolator produces simple
linear-blended frames that satisfy the RIFEInterpolatorLike protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common.image_io import load_rgb
from src.stage3_rife import (
    VOWELS,
    generate_eye_blink_frames,
    generate_mouth_frames,
)

# ============================================================
# Mock interpolator
# ============================================================


@dataclass
class MockRIFEInterpolator:
    """Linear-blend stand-in that satisfies RIFEInterpolatorLike."""

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
            blended = (
                (
                    (1.0 - t) * frame_start.astype(np.float32)
                    + t * frame_end.astype(np.float32)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            frames.append(blended)
        return frames


# ============================================================
# generate_eye_blink_frames
# ============================================================


class TestGenerateEyeBlinkFrames:
    def test_writes_expected_number_of_frames(self, tmp_path: Path) -> None:
        h, w = 16, 16
        eye_open = np.ones((h, w, 3), dtype=np.uint8) * 255
        eye_closed = np.zeros((h, w, 3), dtype=np.uint8)

        interpolator = MockRIFEInterpolator()
        out_dir = tmp_path / "eye"
        paths = generate_eye_blink_frames(
            eye_open=eye_open,
            eye_closed=eye_closed,
            output_dir=out_dir,
            interpolator=interpolator,  # type: ignore[arg-type]
            n_frames=6,
        )

        assert len(paths) == 6
        for idx, path in enumerate(paths, start=1):
            assert path.exists()
            assert path.name == f"frame_{idx:03d}.png"
            loaded = load_rgb(path)
            assert loaded.shape == (h, w, 3)

        # Mock은 한 번만 호출됨
        assert len(interpolator.calls) == 1
        assert interpolator.calls[0] == ((h, w, 3), (h, w, 3), 6)

    def test_first_and_last_frames_match_endpoints(self, tmp_path: Path) -> None:
        h, w = 8, 8
        eye_open = np.ones((h, w, 3), dtype=np.uint8) * 200
        eye_closed = np.ones((h, w, 3), dtype=np.uint8) * 50

        paths = generate_eye_blink_frames(
            eye_open=eye_open,
            eye_closed=eye_closed,
            output_dir=tmp_path / "eye",
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
            n_frames=4,
        )

        first = load_rgb(paths[0])
        last = load_rgb(paths[-1])
        np.testing.assert_array_equal(first, eye_open)
        np.testing.assert_array_equal(last, eye_closed)

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "eye"
        assert not out_dir.exists()

        generate_eye_blink_frames(
            eye_open=np.zeros((4, 4, 3), dtype=np.uint8),
            eye_closed=np.zeros((4, 4, 3), dtype=np.uint8),
            output_dir=out_dir,
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
            n_frames=2,
        )
        assert out_dir.exists()


# ============================================================
# generate_mouth_frames
# ============================================================


class TestGenerateMouthFrames:
    def _vowel_shapes(
        self, h: int, w: int, missing: set[str] | None = None
    ) -> dict[str, np.ndarray]:
        missing = missing or set()
        shapes: dict[str, np.ndarray] = {}
        for i, v in enumerate(VOWELS):
            if v in missing:
                continue
            shapes[v] = np.full((h, w, 3), fill_value=(i + 1) * 40, dtype=np.uint8)
        return shapes

    def test_all_vowels_written(self, tmp_path: Path) -> None:
        h, w = 8, 8
        mouth_closed = np.zeros((h, w, 3), dtype=np.uint8)
        shapes = self._vowel_shapes(h, w)

        interpolator = MockRIFEInterpolator()
        out_dir = tmp_path / "mouth_out"
        result = generate_mouth_frames(
            mouth_closed=mouth_closed,
            mouth_shapes=shapes,
            output_dir=out_dir,
            interpolator=interpolator,  # type: ignore[arg-type]
            n_frames=5,
        )

        assert set(result.keys()) == set(VOWELS)
        # 각 모음당 5개 프레임 + 5모음 → 5회 호출
        assert len(interpolator.calls) == 5
        for vowel in VOWELS:
            vowel_dir = out_dir / f"mouth_{vowel}"
            assert vowel_dir.is_dir()
            paths = result[vowel]
            assert len(paths) == 5
            for p in paths:
                assert p.parent == vowel_dir
                assert p.exists()

    def test_missing_vowel_skipped(self, tmp_path: Path) -> None:
        h, w = 8, 8
        mouth_closed = np.zeros((h, w, 3), dtype=np.uint8)
        shapes = self._vowel_shapes(h, w, missing={"u", "o"})

        result = generate_mouth_frames(
            mouth_closed=mouth_closed,
            mouth_shapes=shapes,
            output_dir=tmp_path / "mouth",
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
            n_frames=3,
        )
        # u, o는 누락 → 결과 dict에 없어야 함
        assert set(result.keys()) == {"a", "i", "e"}
        assert not (tmp_path / "mouth" / "mouth_u").exists()
        assert not (tmp_path / "mouth" / "mouth_o").exists()

    def test_empty_vowel_map_returns_empty_dict(self, tmp_path: Path) -> None:
        result = generate_mouth_frames(
            mouth_closed=np.zeros((4, 4, 3), dtype=np.uint8),
            mouth_shapes={},
            output_dir=tmp_path / "mouth",
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
            n_frames=2,
        )
        assert result == {}

    def test_frame_endpoints_match(self, tmp_path: Path) -> None:
        h, w = 4, 4
        mouth_closed = np.ones((h, w, 3), dtype=np.uint8) * 10
        shapes = {"a": np.ones((h, w, 3), dtype=np.uint8) * 250}

        result = generate_mouth_frames(
            mouth_closed=mouth_closed,
            mouth_shapes=shapes,
            output_dir=tmp_path / "mouth",
            interpolator=MockRIFEInterpolator(),  # type: ignore[arg-type]
            n_frames=3,
        )
        paths = result["a"]
        first = load_rgb(paths[0])
        last = load_rgb(paths[-1])
        np.testing.assert_array_equal(first, mouth_closed)
        np.testing.assert_array_equal(last, shapes["a"])
