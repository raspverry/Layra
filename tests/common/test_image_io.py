"""src/common/image_io.py 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.image_io import (
    compute_iou,
    load_rgb,
    load_rgba,
    save_rgb,
    save_rgba,
)


class TestSaveLoadRoundTrip:
    def test_rgba_round_trip(
        self,
        tmp_out_dir: Path,
        dummy_rgba_64x48,  # type: ignore[no-untyped-def]
    ) -> None:
        """저장 후 로드하면 동일한 배열이 나와야 한다."""
        import numpy as np

        path = tmp_out_dir / "test.png"
        save_rgba(dummy_rgba_64x48, path)

        loaded = load_rgba(path)
        assert loaded.shape == dummy_rgba_64x48.shape
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, dummy_rgba_64x48)

    def test_rgb_round_trip(
        self,
        tmp_out_dir: Path,
        dummy_rgb_64x48,  # type: ignore[no-untyped-def]
    ) -> None:
        import numpy as np

        path = tmp_out_dir / "test_rgb.png"
        save_rgb(dummy_rgb_64x48, path)
        loaded = load_rgb(path)
        assert loaded.shape == dummy_rgb_64x48.shape
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, dummy_rgb_64x48)

    def test_save_creates_parent_dirs(
        self,
        tmp_out_dir: Path,
        dummy_rgba_64x48,  # type: ignore[no-untyped-def]
    ) -> None:
        path = tmp_out_dir / "nested" / "deep" / "img.png"
        save_rgba(dummy_rgba_64x48, path)
        assert path.exists()


class TestValidation:
    def test_save_rgba_rejects_wrong_shape(
        self, tmp_out_dir: Path
    ) -> None:
        import numpy as np

        bad = np.zeros((10, 10, 3), dtype=np.uint8)  # RGB not RGBA
        with pytest.raises(ValueError, match="HxWx4"):
            save_rgba(bad, tmp_out_dir / "x.png")

    def test_save_rgba_rejects_wrong_dtype(
        self, tmp_out_dir: Path
    ) -> None:
        import numpy as np

        bad = np.zeros((10, 10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            save_rgba(bad, tmp_out_dir / "x.png")

    def test_load_rgba_raises_on_missing(self, tmp_out_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_rgba(tmp_out_dir / "does_not_exist.png")


class TestComputeIoU:
    def test_identical_masks(self) -> None:
        import numpy as np

        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        assert compute_iou(mask, mask) == 1.0

    def test_disjoint_masks(self) -> None:
        import numpy as np

        a = np.zeros((32, 32), dtype=bool)
        b = np.zeros((32, 32), dtype=bool)
        a[0:10, 0:10] = True
        b[20:30, 20:30] = True
        assert compute_iou(a, b) == 0.0

    def test_half_overlap(self) -> None:
        """두 10x10 마스크가 5x10만큼 겹칠 때 IoU = 50/(100+100-50) = 1/3."""
        import numpy as np

        a = np.zeros((32, 32), dtype=bool)
        b = np.zeros((32, 32), dtype=bool)
        a[0:10, 0:10] = True
        b[0:10, 5:15] = True

        iou = compute_iou(a, b)
        assert abs(iou - (50 / 150)) < 1e-6

    def test_both_empty_is_one(self) -> None:
        import numpy as np

        empty = np.zeros((8, 8), dtype=bool)
        assert compute_iou(empty, empty) == 1.0

    def test_accepts_integer_masks(self) -> None:
        import numpy as np

        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.zeros((8, 8), dtype=np.uint8)
        a[0:4, :] = 1
        b[0:4, :] = 1
        assert compute_iou(a, b) == 1.0
