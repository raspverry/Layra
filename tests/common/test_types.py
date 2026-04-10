"""src/common/types.py 테스트."""

from __future__ import annotations

from src.common.types import LayerName, LayerSet


class TestLayerName:
    def test_has_all_23_layers(self) -> None:
        """See-Through 논문 기준 23개 레이어가 모두 존재해야 한다."""
        assert len(LayerName) == 23

    def test_head_region_layers(self) -> None:
        head = {
            LayerName.HAIR_FRONT,
            LayerName.HAIR_BACK,
            LayerName.HAIR_SIDE_L,
            LayerName.HAIR_SIDE_R,
            LayerName.FACE,
            LayerName.EYEBROW_L,
            LayerName.EYEBROW_R,
            LayerName.EYE_L,
            LayerName.EYE_R,
            LayerName.EYE_WHITE_L,
            LayerName.EYE_WHITE_R,
            LayerName.NOSE,
            LayerName.MOUTH,
            LayerName.EAR_L,
            LayerName.EAR_R,
        }
        assert len(head) == 15

    def test_body_region_layers(self) -> None:
        body = {
            LayerName.NECK,
            LayerName.BODY,
            LayerName.ARM_L,
            LayerName.ARM_R,
            LayerName.HAND_L,
            LayerName.HAND_R,
            LayerName.CLOTHES,
            LayerName.ACCESSORIES,
        }
        assert len(body) == 8

    def test_string_values_are_snake_case(self) -> None:
        """레이어 이름은 PSD와의 매칭을 위해 snake_case."""
        for name in LayerName:
            assert " " not in name.value
            assert name.value == name.value.replace(" ", "_")


class TestLayerSet:
    def test_empty(self) -> None:
        ls = LayerSet()
        assert len(ls) == 0
        assert LayerName.BODY not in ls
        assert ls.get(LayerName.BODY) is None

    def test_add_and_retrieve(self, dummy_rgba_64x48) -> None:  # type: ignore[no-untyped-def]
        ls = LayerSet()
        ls.layers[LayerName.BODY] = dummy_rgba_64x48
        ls.drawing_order.append(LayerName.BODY)

        assert len(ls) == 1
        assert LayerName.BODY in ls
        assert ls.get(LayerName.BODY) is dummy_rgba_64x48
        assert ls.drawing_order == [LayerName.BODY]
