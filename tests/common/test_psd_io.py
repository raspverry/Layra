"""src/common/psd_io.py 테스트.

`write_psd`는 psd_tools 1.14+의 `PSDImage.new` + `create_pixel_layer`로
per-layer PSD를 작성하며, `parse_psd`와 round-trip 가능하다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.psd_io import _alpha_composite, parse_psd, write_psd
from src.common.types import LayerName, LayerSet


class TestAlphaComposite:
    def test_over_opaque_fully_opaque(self) -> None:
        """불투명 레이어를 불투명 레이어 위에 올리면 결과는 위 레이어."""
        import numpy as np

        base = np.zeros((4, 4, 4), dtype=np.uint8)
        base[..., 0] = 255  # 빨강
        base[..., 3] = 255

        overlay = np.zeros((4, 4, 4), dtype=np.uint8)
        overlay[..., 2] = 255  # 파랑
        overlay[..., 3] = 255

        out = _alpha_composite(base, overlay)
        assert out[0, 0, 0] == 0  # 빨강 채널 0
        assert out[0, 0, 2] == 255  # 파랑 채널 255
        assert out[0, 0, 3] == 255  # 완전 불투명

    def test_over_transparent_preserves_base(self) -> None:
        """완전 투명한 overlay를 올리면 base가 그대로."""
        import numpy as np

        base = np.zeros((4, 4, 4), dtype=np.uint8)
        base[..., 0] = 255
        base[..., 3] = 255

        overlay = np.zeros((4, 4, 4), dtype=np.uint8)  # alpha=0

        out = _alpha_composite(base, overlay)
        np.testing.assert_array_equal(out, base)

    def test_shape_mismatch_raises(self) -> None:
        import numpy as np

        a = np.zeros((4, 4, 4), dtype=np.uint8)
        b = np.zeros((8, 8, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            _alpha_composite(a, b)


class TestWritePSD:
    def test_empty_layerset_raises(self, tmp_out_dir: Path) -> None:
        ls = LayerSet()
        with pytest.raises(ValueError, match="empty LayerSet"):
            write_psd(ls, tmp_out_dir / "empty.psd")

    def test_writes_single_layer(
        self,
        tmp_out_dir: Path,
        dummy_rgba_64x48,  # type: ignore[no-untyped-def]
    ) -> None:
        ls = LayerSet()
        ls.layers[LayerName.BODY] = dummy_rgba_64x48
        ls.drawing_order.append(LayerName.BODY)

        out = tmp_out_dir / "single.psd"
        written = write_psd(ls, out)
        assert written == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_writes_multiple_layers(
        self,
        tmp_out_dir: Path,
        dummy_rgba_64x48,  # type: ignore[no-untyped-def]
    ) -> None:
        import numpy as np

        ls = LayerSet()
        ls.layers[LayerName.BODY] = dummy_rgba_64x48

        hair = np.zeros_like(dummy_rgba_64x48)
        hair[:20, :, 2] = 255  # 상단 파랑
        hair[:20, :, 3] = 255
        ls.layers[LayerName.HAIR_FRONT] = hair

        ls.drawing_order = [LayerName.BODY, LayerName.HAIR_FRONT]

        out = tmp_out_dir / "multi.psd"
        write_psd(ls, out)
        assert out.exists()


class TestPSDRoundTrip:
    def test_roundtrip_preserves_layer_names(
        self,
        tmp_out_dir: Path,
        dummy_rgba_64x48,  # type: ignore[no-untyped-def]
    ) -> None:
        """write_psd → parse_psd 후 레이어 이름과 순서가 보존되어야 한다."""
        import numpy as np

        ls = LayerSet()
        ls.layers[LayerName.BODY] = dummy_rgba_64x48
        hair = np.zeros_like(dummy_rgba_64x48)
        hair[:20, :, 2] = 255
        hair[:20, :, 3] = 255
        ls.layers[LayerName.HAIR_FRONT] = hair
        ls.drawing_order = [LayerName.BODY, LayerName.HAIR_FRONT]

        out = tmp_out_dir / "roundtrip.psd"
        write_psd(ls, out)

        parsed = parse_psd(out)
        assert LayerName.BODY in parsed
        assert LayerName.HAIR_FRONT in parsed
        assert len(parsed) == 2
        # 형상과 dtype이 일치하는지
        assert parsed.get(LayerName.BODY).shape == dummy_rgba_64x48.shape  # type: ignore[union-attr]
        assert parsed.get(LayerName.BODY).dtype == np.uint8  # type: ignore[union-attr]
