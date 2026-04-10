"""src/stage2_sam3/psd_parser.py 테스트."""

from __future__ import annotations

from src.common.types import LayerName, LayerSet
from src.stage2_sam3.psd_parser import extract_body_parts, layerset_to_body_parts


class TestLayersetToBodyParts:
    def test_all_parts_present(
        self, dummy_rgba_64x48  # type: ignore[no-untyped-def]
    ) -> None:
        import numpy as np

        ls = LayerSet()
        ls.layers[LayerName.BODY] = dummy_rgba_64x48
        ls.layers[LayerName.HAIR_FRONT] = dummy_rgba_64x48 * 0
        ls.layers[LayerName.HAIR_BACK] = dummy_rgba_64x48
        ls.layers[LayerName.MOUTH] = dummy_rgba_64x48
        ls.layers[LayerName.NECK] = dummy_rgba_64x48

        parts = layerset_to_body_parts(ls)
        assert parts.body is not None
        assert parts.hair_front is not None
        assert parts.hair_back is not None
        assert parts.mouth is not None
        assert parts.neck is not None

        assert np.array_equal(parts.body, dummy_rgba_64x48)

    def test_missing_parts_return_none(self) -> None:
        ls = LayerSet()
        parts = layerset_to_body_parts(ls)
        assert parts.body is None
        assert parts.hair_front is None
        assert parts.hair_back is None
        assert parts.mouth is None
        assert parts.neck is None

    def test_extract_body_parts_tuple(
        self, dummy_rgba_64x48  # type: ignore[no-untyped-def]
    ) -> None:
        ls = LayerSet()
        ls.layers[LayerName.BODY] = dummy_rgba_64x48
        ls.layers[LayerName.HAIR_FRONT] = dummy_rgba_64x48

        body, hair_front, hair_back = extract_body_parts(ls)
        assert body is not None
        assert hair_front is not None
        assert hair_back is None
